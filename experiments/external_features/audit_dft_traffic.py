"""Audit whether the official DfT London count file is dense enough for daily modelling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OFFICIAL_URL = (
    "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/"
    "rawcount/region_id/dft_rawcount_region_id_6.csv"
)


def coverage_table(frame: pd.DataFrame, start_year: int = 2021, end_year: int = 2024) -> list[dict]:
    required = {"year", "count_date", "count_point_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing DfT columns: {sorted(missing)}")
    data = frame[frame.year.between(start_year, end_year)].copy()
    data["count_date"] = pd.to_datetime(data.count_date, errors="raise")
    rows = []
    for year in range(start_year, end_year + 1):
        part = data[data.year == year]
        expected_days = 366 if pd.Timestamp(year, 12, 31).dayofyear == 366 else 365
        observed_days = int(part.count_date.nunique())
        rows.append(
            {
                "year": year,
                "rows": int(len(part)),
                "unique_survey_days": observed_days,
                "expected_calendar_days": expected_days,
                "calendar_coverage_fraction": observed_days / expected_days,
                "unique_count_points": int(part.count_point_id.nunique()),
                "first_date": None if part.empty else str(part.count_date.min().date()),
                "last_date": None if part.empty else str(part.count_date.max().date()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external_features/traffic/dft_rawcount_region_6.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external_features/traffic/coverage_audit.json"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input, usecols=["year", "count_date", "count_point_id"])
    years = coverage_table(frame)
    minimum_daily_coverage = 0.90
    audit = {
        "source": "DfT Road Traffic Statistics, London raw manual counts",
        "official_url": OFFICIAL_URL,
        "decision": "reject_as_daily_feature",
        "reason": (
            "The file contains one-day manual surveys, not a continuous daily London traffic series. "
            "Using interpolation across the long gaps would invent the signal being tested."
        ),
        "minimum_required_calendar_coverage": minimum_daily_coverage,
        "passes_daily_coverage_gate": all(
            row["calendar_coverage_fraction"] >= minimum_daily_coverage for row in years
        ),
        "years": years,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
