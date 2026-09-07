# Research protocol and interpretation

## Task

Aitken estimates daily PM2.5 over Greater London on a 1 km grid. Inputs combine
weather, satellite and static predictors with strictly earlier-day measured
PM2.5 history. Same-day PM2.5 measurements are not predictor inputs.

Environmental products include reanalysis and quarterly satellite composites.
A same-quarter composite may contain imagery acquired later in the quarter.
The completed experiments therefore represent **retrospective mapping and
nowcasting**, rather than a demonstrated forecast issued in real time.

## Models

- **HGB:** histogram gradient boosting, blending raw- and log-target models.
- **Limited-monotonic HGB:** a separate HGB configuration with constraints on
  three prior-PM2.5 history variables.
- **XGBoost:** an independently fitted gradient-boosted tree baseline.
- **ANN:** a five-seed ensemble of sklearn multilayer perceptrons.
- **Hybrid HGB + residual U-Net:** 66 predictor channels plus an HGB baseline
  map; a three-seed ensemble learns a spatial correction in standardised
  log-PM2.5 space, with a frozen correction weight of 0.425. Concentration is
  obtained after inverse transformation.
- **Standalone U-Net:** direct estimation from 66 channels, without HGB input,
  residual targets or baseline addition. Reported as the pre-specified seed-42
  model and an equal-weight three-seed ensemble.

## Common evaluation

| Item | Protocol |
|---|---|
| Final fitting period | 1 January 2021–31 December 2023 |
| Requested evaluation period | Calendar year 2024 |
| Available evaluation observations | 1 January–14 December 2024 |
| Evaluation size | 9,619 LAQN station-days across 349 dates |
| Primary comparison metric | RMSE in µg/m³ |
| Standalone single-model seed | 42, pre-specified |
| Standalone ensemble seeds | 42, 11 and 22 |

The year 2024 had been inspected in earlier project experiments. This is a
retrospective common benchmark, not an untouched confirmatory test.

## What the results support

The hybrid has RMSE 3.7765 µg/m³, compared with 3.7920 for unconstrained HGB.
The final 50,000-repeat paired calendar-day bootstrap gives a descriptive
95% interval of approximately [-0.00003, 0.0283] µg/m³ for HGB-minus-hybrid
RMSE. Because the interval includes zero, this does not establish decisive
hybrid superiority. It also does not establish statistical equivalence.
The intervals are post-selection and unadjusted, rather than confirmatory tests.

The standalone ensemble has RMSE 4.1227; the pre-specified single network has
RMSE 4.2816. The best individual seed observed on 2024 is not substituted for
the pre-specified single-model result.

The [result table](../results/metrics.csv) contains aggregate metrics derived
from the same saved station-prediction rows. See [provenance](../results/README.md).

## Limits and follow-up work

The current evaluation measures performance at monitoring locations, not every
map cell. Rare high-pollution values remain underpredicted. Residual training
used in-sample HGB maps; a stronger design would use out-of-fold maps.
Same-site history and dependencies between observation sources also warrant
further evaluation. Generalisation to another city has not been established.

Planned research includes an untouched time period, station-held-out evaluation,
out-of-fold baseline maps and uncertainty analysis. None of these planned
experiments is counted as a completed result in this repository.
