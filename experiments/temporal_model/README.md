# Temporal hybrid PM2.5 model

This experiment compares the current HGB + residual U-Net hybrid with the same
model after adding five past-only time-series channels:

- 3-day mean
- 7-day mean
- 7-day maximum
- 7-day standard deviation
- 3-day trend

All five channels come from the existing `pm25_lag1_idw_causal` map. Because
that source already ends at the previous day, the rolling features contain no
same-day PM2.5 label and no future information.

## Models

| Name | Inputs |
|---|---|
| `A_current_hybrid` | 66 predictors + matched HGB map |
| `B_temporal` | The same inputs + five past-only temporal channels |

Both models use the same residual U-Net, loss, schedule and seeds. This isolates
the effect of the temporal features.

## Evaluation

The first screen fits on 2021–2022 and validates on 2023. The final comparison
fits both models from scratch on 2021–2023 with seeds 42, 11 and 22, ensembles
their predictions, and scores the same 9,619 LAQN observations in 2024.

The 2024 comparison is retrospective because this project had already examined
2024. It did not select features or model settings.

## Run

```bash
python -m unittest discover -s experiments/temporal_model -p "test_*.py" -v

python -u experiments/temporal_model/run_experiment.py \
  --data-dir pm25_london_bundle/data/processed/london_1km_daily \
  --reference-bundle pm25_london_bundle \
  --history-feature pm25_lag1_idw_causal \
  --output-dir artifacts/temporal_model \
  --gpu-hours 4 \
  --evaluate-2024
```

Large predictions, fitted checkpoints and source data remain local. The curated
[result report](../../results/temporal_model_20260912/REPORT.md) contains the
verified aggregate statistics.
