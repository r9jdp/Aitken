"""Download and prepare official regional AURN PM2.5 data for 2021-2024."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from aurn import build_daily_features, compile_daily, load_station_catalogue, select_regional_sites


CATALOGUE_URL = "https://uk-air.defra.gov.uk/ajax/import_map_data?doajax=true&n=aurn"
CSV_URL = "https://uk-air.defra.gov.uk/datastore/data_files/site_data/{site}_{year}.csv?v=1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(session: requests.Session, url: str, path: Path) -> dict:
    if path.exists() and path.stat().st_size > 100:
        return {"url": url, "path": str(path), "sha256": sha256(path), "status": "cached"}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            response = session.get(url, timeout=90)
            response.raise_for_status()
            temporary.write_bytes(response.content)
            if temporary.stat().st_size < 100:
                raise ValueError("download was unexpectedly small")
            temporary.replace(path)
            return {"url": url, "path": str(path), "sha256": sha256(path), "status": "downloaded"}
        except Exception:
            if temporary.exists():
                temporary.unlink()
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external_features/aurn"))
    parser.add_argument("--data-dir", type=Path, default=Path("pm25_london_bundle/data/processed/london_1km_daily"))
    parser.add_argument("--min-km", type=float, default=30.0)
    parser.add_argument("--max-km", type=float, default=110.0)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 Aitken-PM25-research/1.0"})

    catalogue = output / "aurn_sites.json"
    catalogue_record = download(session, CATALOGUE_URL, catalogue)
    sites = select_regional_sites(load_station_catalogue(catalogue), args.min_km, args.max_km)
    sites.to_csv(output / "selected_sites.csv", index=False)

    records = []
    years = list(range(2021, 2025))
    for site in sites.itertuples(index=False):
        for year in years:
            url = CSV_URL.format(site=site.site_id, year=year)
            path = raw / f"{site.site_id}_{year}.csv"
            try:
                records.append(download(session, url, path))
            except requests.RequestException as error:
                # Newer stations can legitimately lack earlier study years;
                # transient server failures are recorded so rerunning can fill
                # them without losing already downloaded files.
                records.append({"url": url, "path": str(path), "status": "unavailable", "error": str(error)})

    daily = compile_daily(raw, sites, years)
    daily.to_csv(output / "aurn_daily.csv", index=False)
    metadata = json.loads((args.data_dir / "metadata.json").read_text())
    feature_map = np.load(args.data_dir / "features_float16.npy", mmap_mode="r")
    mask = np.load(args.data_dir / "gla_mask.npy") > 0
    names = metadata["feature_names"]
    u_index, v_index = names.index("era5_u10"), names.index("era5_v10")
    u10 = np.asarray(feature_map[:, u_index, mask], np.float32).mean(axis=1)
    v10 = np.asarray(feature_map[:, v_index, mask], np.float32).mean(axis=1)
    values, feature_names, audit = build_daily_features(daily, u10, v10)
    np.savez_compressed(output / "aurn_features.npz", values=values, names=np.asarray(feature_names))
    audit.to_csv(output / "daily_availability.csv", index=False)

    coverage = (
        daily.groupby("site_id")
        .agg(first_date=("date", "min"), last_date=("date", "max"), valid_days=("date", "nunique"), median_hours=("valid_hours", "median"))
        .reset_index()
        .merge(sites[["site_id", "site_name", "distance_km", "bearing_deg"]], on="site_id", how="left")
    )
    coverage.to_csv(output / "coverage.csv", index=False)
    manifest = {
        "source": "DEFRA UK-AIR AURN",
        "catalogue_url": CATALOGUE_URL,
        "catalogue": catalogue_record,
        "years": years,
        "selection": {"minimum_km": args.min_km, "maximum_km": args.max_km, "background_only": True},
        "selected_site_count": int(len(sites)),
        "usable_site_count": int(daily.site_id.nunique()),
        "usable_station_days": int(len(daily)),
        "feature_names": feature_names,
        "downloads": records,
        "quality_rule": "0-500 ug/m3, ratified hourly rows, at least 18 valid hours per station-day",
        "timing": {
            "same_day": "retrospective mapping/nowcast only",
            "lag1": "strictly prior-day alternative",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("selected_site_count", "usable_site_count", "usable_station_days", "feature_names")}, indent=2))


if __name__ == "__main__":
    main()
