# Data sources

Aitken's first case study uses downloaded monitoring observations and
environmental products for Greater London, 2021–2024. The London experiments
use no synthetic PM2.5 training labels.

| Source | Role in the study | Provider |
|---|---|---|
| London Air Quality Network (LAQN) | Regulatory PM2.5 observations and primary evaluation | [LondonAir](https://www.londonair.org.uk/) |
| Breathe London | Community-sensor PM2.5 observations | [Breathe London](https://www.breathelondon.org/) |
| ERA5 | Weather covariates | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) |
| Sentinel-5P | NO2, CO, aerosol index and cloud fraction | [Copernicus Data Space](https://dataspace.copernicus.eu/) |
| Sentinel-2 L2A | Quarterly NDVI and NDBI composites | [Copernicus Data Space](https://dataspace.copernicus.eu/) |
| Census 2021 | Population covariates | [Office for National Statistics](https://www.ons.gov.uk/census) |
| Ordnance Survey | Roads, greenspace and elevation | [Ordnance Survey](https://www.ordnancesurvey.co.uk/) |
| London Atmospheric Emissions Inventory 2022 | Emissions covariates | [London Datastore](https://data.london.gov.uk/) |
| Greater London boundary | Study-domain mask | [London Datastore](https://data.london.gov.uk/) |

NASA FIRMS fire data were investigated during development. Five fire/smoke
channels were excluded from the final 66-feature models.

## Preparation

Hourly observations outside 0–500 µg/m³ are rejected, duplicate station-hours
are removed, and a station-day requires at least 18 valid hourly values.
Observations and environmental predictors are aligned to an EPSG:27700 grid
at 1 km resolution, with 1,719 in-domain cells in a 48 × 64 tensor.

The prepared study package contains 444,202 station-days: 40,949 from LAQN and
403,253 from Breathe London. These are counts across the preparation period,
not the size of the regulatory test set. Final model evaluation uses the same
9,619 available 2024 LAQN station-days for every reported model.

Gridding, interpolation, imputation and model estimates are distinct from
observed measurements. Earlier-day PM2.5 interpolation supplies predictors;
it does not supply fabricated training labels at unmonitored locations.

## Access and redistribution

This initial repository contains documentation, aggregate model metrics and
selected figures. Raw observations, prepared arrays, station-level prediction
tables and checkpoints are not included. Some acquisition workflows require
provider accounts or API credentials.

Acquisition/preparation scripts and a fuller data-access guide are planned
for a subsequent code release. Redistribution permissions and attribution
requirements will be reviewed before publishing any downloadable data sample.
Provider access does not automatically grant redistribution rights.
