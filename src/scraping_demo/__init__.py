from pathlib import Path
import sys

import polars as pl


DATE = "Flight Date"
AIRPORT = "Airport Name"
DAMAGE = "Effect Amount of damage"
SPEED = "Speed IAS in knots"
AIRLINE = "Aircraft Airline Operator"


def load_strike_reports() -> pl.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    return pl.read_csv(project_root / "birdstrikes.csv", try_parse_dates=True)


def prepare_strike_reports(reports: pl.DataFrame) -> pl.DataFrame:
    return reports.filter(pl.col(DATE).dt.year() > 1990).select(
        DATE, AIRPORT, DAMAGE, SPEED, AIRLINE
    )


def _airport_damage_window(reports: pl.DataFrame, start_year: int) -> pl.DataFrame:
    damage_value = pl.col(DAMAGE).str.strip_chars()
    known_damage = damage_value.is_not_null() & (damage_value != "")
    return (
        reports.filter(pl.col(DATE).dt.year().is_between(start_year, start_year + 2))
        .group_by(AIRPORT)
        .agg(
            pl.len().alias("reports"),
            known_damage.sum().alias("known_damage_reports"),
            (known_damage & (damage_value != "None")).sum().alias("damaged_reports"),
        )
        .with_columns(
            pl.when(pl.col("known_damage_reports") > 0)
            .then(pl.col("damaged_reports") / pl.col("known_damage_reports") * 100)
            .otherwise(None)
            .alias("damage_percentage")
        )
    )


def rank_airports_with_damage(reports: pl.DataFrame) -> pl.DataFrame:
    last_year = reports.get_column(DATE).dt.year().max()
    increases = []
    for start_year in range(1991, last_year - 4, 6):
        previous = _airport_damage_window(reports, start_year).select(
            AIRPORT,
            pl.col("reports").alias("First Window Reports"),
            pl.col("damage_percentage").alias("First Window Damage %"),
        )
        following = _airport_damage_window(reports, start_year + 3).select(
            AIRPORT,
            pl.col("reports").alias("Next Window Reports"),
            pl.col("damage_percentage").alias("Next Window Damage %"),
        )
        increases.append(
            previous.join(following, on=AIRPORT, how="inner")
            .filter(
                (pl.col("First Window Reports") >= 10)
                & (pl.col("Next Window Reports") >= 10)
                & (pl.col("Next Window Damage %") > pl.col("First Window Damage %"))
            )
            .with_columns(
                pl.lit(f"{start_year}-{start_year + 2}").alias("First Window"),
                pl.lit(f"{start_year + 3}-{start_year + 5}").alias("Next Window"),
                (pl.col("Next Window Damage %") - pl.col("First Window Damage %")).alias(
                    "Increase (percentage points)"
                ),
            )
        )
    return pl.concat(increases).select(
        AIRPORT,
        "First Window",
        "First Window Reports",
        "First Window Damage %",
        "Next Window",
        "Next Window Reports",
        "Next Window Damage %",
        "Increase (percentage points)",
    ).sort("Increase (percentage points)", descending=True)


def damage_by_velocity(reports: pl.DataFrame) -> pl.DataFrame:
    damage_value = pl.col(DAMAGE).str.strip_chars()
    known_damage = damage_value.is_not_null() & (damage_value != "")
    grouped = (
        reports.filter(pl.col(SPEED).is_not_null() & known_damage)
        .with_columns(((pl.col(SPEED) // 50) * 50).alias("_speed_start"))
        .group_by(["_speed_start", DAMAGE])
        .agg(pl.len().alias("count"))
    )
    totals = grouped.group_by("_speed_start").agg(pl.col("count").sum().alias("total"))
    percentages = (
        grouped.join(totals, on="_speed_start")
        .with_columns((pl.col("count") / pl.col("total") * 100).alias("percentage"))
        .with_columns(
            pl.concat_str(
                [
                    pl.col("_speed_start").cast(pl.String),
                    pl.lit("-"),
                    (pl.col("_speed_start") + 49).cast(pl.String),
                ]
            ).alias("Speed Range")
        )
    )
    categories = percentages.get_column(DAMAGE).unique().sort().to_list()
    return (
        percentages.pivot(
            on=DAMAGE,
            on_columns=categories,
            index=["_speed_start", "Speed Range"],
            values="percentage",
        )
        .sort("_speed_start")
        .drop("_speed_start")
    )


def top_5_airlines_per_year(reports: pl.DataFrame) -> pl.DataFrame:
    return (
        reports.with_columns(pl.col(DATE).dt.year().alias("Year"))
        .group_by(["Year", AIRLINE])
        .agg(pl.len().alias("Reports"))
        .with_columns(
            pl.col("Reports")
            .rank(method="min", descending=True)
            .over("Year")
            .alias("Rank")
        )
        .filter(pl.col("Rank") <= 5)
        .sort(["Year", "Rank", AIRLINE])
    )


def main() -> None:
    reports = prepare_strike_reports(load_strike_reports())
    results = {
        "airports_with_damage_increase": rank_airports_with_damage(reports),
        "damage_by_velocity": damage_by_velocity(reports),
        "top_5_airlines_per_year": top_5_airlines_per_year(reports),
    }

    output_directory = Path(__file__).resolve().parents[2] / "output"
    output_directory.mkdir(exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    for name, result in results.items():
        result.write_csv(output_directory / f"{name}.csv")
        print(f"{name}:")
        print(result)
