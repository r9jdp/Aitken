# Short explanation for the presentation

We kept the same U-Net architectures and tested one small preprocessing change: capping only the highest 0.5% or 1% of training targets. This is called winsorisation. We did not replace all readings above 25 with the median, and we did not change any actual validation or test observations.

For the standalone U-Net, the selected cap lowered 2023 validation RMSE from 3.2897 to 3.2348 µg/m³. We then tested matched three-seed ensembles on the original 2024 observations.
The 2024 RMSE was 4.0592 without capping and 4.1435 with capping. The corresponding R² values were 0.4430 and 0.4196. High-pollution MAE changed from 15.9819 to 16.7066 µg/m³.
The paired seven-day-block 95% interval for the RMSE change was [+0.0016, +0.1790] µg/m³. The interval is above zero, supporting worse RMSE with capping in this retrospective comparison.
Recommendation: do not adopt this cap. The validation improvement did not carry over consistently to 2024; keep the current model.

For the hybrid U-Net, neither cap improved RMSE without worsening overall or high-pollution MAE on validation, so we kept the existing model and did not perform an additional 2024 trial for it.

High pollution is not automatically bad data. This experiment tests whether mild training-target capping helps; it does not prove the extreme observations were wrong. The project concept, dashboard and original presentation results remain unchanged. Any 2024 comparison is retrospective because that year was examined earlier.
