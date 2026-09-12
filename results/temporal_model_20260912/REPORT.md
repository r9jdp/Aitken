# Temporal hybrid model results

## Result

Adding five past-only temporal features gave a small, consistent improvement
over the freshly trained matched hybrid control on the retrospective 2024 data.
It is the best balanced experimental version, but the published hybrid remains
the official model because the temporal version missed the stricter 2023
positive-bias guard by 0.0132 µg/m³.

## 2024 comparison

Both versions were trained from scratch on 2021–2023 as three-seed ensembles and
tested on the same 9,619 LAQN station-days across 349 labelled dates in 2024.

| Model | MAE | RMSE | R² | MAE below 15 | Bias below 15 | MAE at or above 25 |
|---|---:|---:|---:|---:|---:|---:|
| Matched current hybrid | 2.3443 | 3.7815 | 0.5166 | 1.7239 | -0.0601 | 14.1090 |
| Temporal hybrid | **2.3361** | **3.7679** | **0.5201** | **1.7174** | -0.0549 | **14.0490** |

Against the matched control, the temporal model changed:

- MAE by -0.0082 µg/m³
- RMSE by -0.0136 µg/m³
- R² by +0.0035
- MAE below 15 by -0.0066 µg/m³
- MAE at or above 25 by -0.0600 µg/m³

The paired seven-day block-bootstrap interval for temporal-minus-control RMSE
was [-0.0203, -0.0075] µg/m³. Negative values favour the temporal model. This
interval is descriptive because 2024 was not an untouched test set.

## 2023 screen

Both models were fitted on 2021–2022 and evaluated on the same 11,102 LAQN
station-days in 2023.

| Model | MAE | RMSE | R² | MAE below 15 | Bias below 15 | MAE at or above 25 |
|---|---:|---:|---:|---:|---:|---:|
| Matched current hybrid | 2.1020 | 3.1409 | 0.6933 | 1.7675 | +0.3892 | 6.9648 |
| Temporal hybrid | **2.1004** | **3.1401** | **0.6935** | **1.7664** | +0.4024 | **6.8934** |

The temporal model improved the error metrics, but normal-range bias became
0.0132 µg/m³ more positive. Therefore it did not pass the pre-written promotion
rule and has not replaced the published model.

## Integrity

- The source was `pm25_lag1_idw_causal`.
- No same-day or future PM2.5 was used as an input.
- No observations were capped, smoothed or changed.
- Source inputs remained byte-for-byte unchanged.
- The audit recalculated both models' statistics on identical 2024 rows and
  verified all six final checkpoint hashes.

[Code and run instructions](../../experiments/temporal_model/README.md) ·
[Independent audit](audit.json) · [Exact metrics](metrics.csv)
