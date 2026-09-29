"""Prepare regional AURN PM2.5 signals without changing the frozen study cube."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


LONDON_LAT = 51.5074
LONDON_LON = -0.1278
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2024-12-31")


def load_station_catalogue(path: Path) -> list[dict]:
    """Read the small JavaScript assignment returned by DEFRA's map endpoint."""
    text = Path(path).read_text(encoding="utf-8").strip()
    prefix = "markers = "
    if not text.startswith(prefix):
        raise ValueError("Unexpected AURN station-catalogue format")
    payload = json.loads(text[len(prefix) :])
    stations = payload.get("aurn")
    if not isinstance(stations, list):
        raise ValueError("AURN station catalogue has no station list")
    return stations


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 6371.0088 * 2.0 * math.asin(math.sqrt(a))


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    east = math.sin(dlambda) * math.cos(phi2)
    north = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.degrees(math.atan2(east, north)) % 360.0


def select_regional_sites(
    stations: list[dict], minimum_km: float = 30.0, maximum_km: float = 110.0
) -> pd.DataFrame:
    """Choose non-traffic PM2.5 sites around, but not inside, Greater London."""
    records = []
    for station in stations:
        if "PM25" not in station.get("parameter_ids", []):
            continue
        # DEFRA IDs 217 and 271 are background and rural sites. Traffic sites
        # are intentionally excluded so the signal represents regional episodes.
        if str(station.get("environment_id")) not in {"217", "271"}:
            continue
        latitude = float(station["latitude"])
        longitude = float(station["longitude"])
        distance = haversine_km(LONDON_LAT, LONDON_LON, latitude, longitude)
        if not minimum_km <= distance <= maximum_km:
            continue
        records.append(
            {
                "site_id": station["site_id"],
                "uka_id": station["uka_id"],
                "site_name": station["site_name"],
                "latitude": latitude,
                "longitude": longitude,
                "distance_km": distance,
                "bearing_deg": bearing_degrees(LONDON_LAT, LONDON_LON, latitude, longitude),
                "environment_id": str(station["environment_id"]),
            }
        )
    result = pd.DataFrame.from_records(records).sort_values(["distance_km", "site_id"])
    if len(result) < 6:
        raise ValueError(f"Only {len(result)} suitable regional AURN sites were found")
    return result.reset_index(drop=True)


def _normalise_header(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", str(value)).lower()
    return re.sub(r"[^a-z0-9.]+", "", value)


def parse_hourly_csv(path: Path, site: pd.Series, year: int) -> pd.DataFrame:
    """Convert one official DEFRA site-year CSV to quality-controlled daily PM2.5."""
    frame = pd.read_csv(path, skiprows=4, low_memory=False)
    normalised = [_normalise_header(column) for column in frame.columns]
    candidates = [
        index
        for index, name in enumerate(normalised)
        if "pm2.5particulatematterhourlymeasured" in name
    ]
    if not candidates:
        return pd.DataFrame()
    pm_index = candidates[0]
    pm_column = frame.columns[pm_index]
    status_column = frame.columns[pm_index + 1] if pm_index + 1 < len(frame.columns) else None
    if _normalise_header(frame.columns[0]) != "date" or _normalise_header(frame.columns[1]) != "time":
        raise ValueError(f"Unexpected date columns in {path}")

    date = pd.to_datetime(frame.iloc[:, 0], format="%d-%m-%Y", errors="coerce")
    values = pd.to_numeric(frame[pm_column], errors="coerce")
    valid = date.notna() & values.between(0.0, 500.0, inclusive="both")
    if status_column is not None and _normalise_header(status_column).startswith("status"):
        status = frame[status_column].astype(str).str.strip().str.upper()
        # Historical study years should now be ratified. Reject provisional and
        # unknown rows rather than mixing quality levels across years.
        valid &= status.eq("R")
    compact = pd.DataFrame({"date": date[valid], "pm25": values[valid]})
    if compact.empty:
        return compact
    daily = compact.groupby("date", as_index=False).agg(pm25=("pm25", "mean"), valid_hours=("pm25", "size"))
    daily = daily[daily.valid_hours >= 18].copy()
    daily["site_id"] = site.site_id
    daily["site_name"] = site.site_name
    daily["latitude"] = float(site.latitude)
    daily["longitude"] = float(site.longitude)
    daily["distance_km"] = float(site.distance_km)
    daily["bearing_deg"] = float(site.bearing_deg)
    daily["year"] = int(year)
    return daily


def compile_daily(raw_directory: Path, sites: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    pieces = []
    for site in sites.itertuples(index=False):
        site_series = pd.Series(site._asdict())
        for year in years:
            path = Path(raw_directory) / f"{site.site_id}_{year}.csv"
            if path.exists() and path.stat().st_size:
                daily = parse_hourly_csv(path, site_series, year)
                if not daily.empty:
                    pieces.append(daily)
    if not pieces:
        raise ValueError("No usable AURN PM2.5 observations were found")
    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(["date", "site_id"]).reset_index(drop=True)


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[good], weights=weights[good])) if good.any() else float("nan")


def build_daily_features(
    daily: pd.DataFrame,
    u10: np.ndarray,
    v10: np.ndarray,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Build day-level regional and wind-aligned signals.

    Same-day features are valid for this project's retrospective nowcast scope.
    Explicit lag-1 copies are also supplied for future forecast-safe comparisons.
    Missing days are filled causally and accompanied by availability/count fields.
    """
    dates = pd.date_range(start, end, freq="D")
    if len(u10) != len(dates) or len(v10) != len(dates):
        raise ValueError("Wind series and study date range do not match")
    grouped = {pd.Timestamp(day): frame for day, frame in daily.groupby("date")}
    raw_rows = []
    for index, day in enumerate(dates):
        frame = grouped.get(day)
        if frame is None or frame.empty:
            raw_rows.append((np.nan, np.nan, np.nan, 0.0, 0.0))
            continue
        values = frame.pm25.to_numpy(float)
        distances = frame.distance_km.to_numpy(float)
        bearings = frame.bearing_deg.to_numpy(float)
        base_weights = 1.0 / np.maximum(distances, 25.0)
        regional_mean = _weighted(values, base_weights)
        regional_max = float(np.nanmax(values))

        # ERA5 u/v describe where the air is moving. Negating both gives the
        # compass direction from which the air reached London.
        source_bearing = math.degrees(math.atan2(-float(u10[index]), -float(v10[index]))) % 360.0
        difference = np.radians((bearings - source_bearing + 180.0) % 360.0 - 180.0)
        alignment = np.maximum(np.cos(difference), 0.0) ** 2
        upwind_mean = _weighted(values, base_weights * alignment)
        if not np.isfinite(upwind_mean):
            upwind_mean = regional_mean
        raw_rows.append((regional_mean, regional_max, upwind_mean, float(len(frame)), 1.0))

    raw = pd.DataFrame(
        raw_rows,
        index=dates,
        columns=["regional_mean", "regional_max", "upwind_mean", "valid_site_count", "available"],
    )
    # Causal fill: at day d this uses only values observed on or before d.
    filled = raw[["regional_mean", "regional_max", "upwind_mean"]].ffill(limit=3).fillna(0.0)
    same = np.column_stack(
        [
            filled.regional_mean,
            filled.regional_max,
            filled.upwind_mean,
            np.log1p(raw.valid_site_count),
            raw.available,
        ]
    ).astype(np.float32)
    lag = np.vstack([np.zeros((1, 3), np.float32), same[:-1, :3]])
    values = np.column_stack([same, lag]).astype(np.float32)
    names = [
        "aurn_regional_mean_same_day",
        "aurn_regional_max_same_day",
        "aurn_upwind_mean_same_day",
        "aurn_log_valid_site_count",
        "aurn_same_day_available",
        "aurn_regional_mean_lag1",
        "aurn_regional_max_lag1",
        "aurn_upwind_mean_lag1",
    ]
    audit = raw.reset_index(names="date")
    return values, names, audit
