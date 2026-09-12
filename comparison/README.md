# London model comparison and high pollution error analysis

Report date: **12 September 2026**. This folder documents the existing product; it does not replace or retrain it.

- [Read the report on GitHub](REPORT.md)
- [Download the Word document](London_PM25_Comparison_and_Error_Analysis.docx)
- [Recomputed error evidence](evidence/high_pollution_diagnosis.json)
- [Published sources and table locators](sources.json)
- [Report review and limitations](REVIEW.md)

## Main findings

The original hybrid HGB + residual U-Net has the best numerical release result: MAE **2.3405**, RMSE **3.7765 µg/m³**, R² **0.5179**. Its advantage over HGB alone is small and not decisive under the saved paired bootstrap.

In the unchanged 2024 benchmark, **216 of 9,619 observations** have PM2.5 ≥25 µg/m³. They represent **2.25%** of observations but **38.76%** of squared error, and all were underpredicted. High concentration alone is not proof of bad data.

The prior controlled capping experiment did not support adoption. The original implementation, model table, dashboard and presentation remain unchanged. Published papers are context, not models evaluated on the same benchmark.

## Reproduce

From the Aitken repository, use Python with `python-docx` and `Pillow`. No training packages or virtual environment are needed for report generation. The document was built using the Codex bundled Python, not the model-training environment.

The report can be rebuilt entirely from the tracked aggregates:

```powershell
python comparison/build_report.py
python -m unittest discover -s comparison -p test_report.py -v
```

To recompute the aggregate diagnosis from the existing local original bundle, pass the saved prediction file explicitly. The script verifies its contents against the release metrics and checks its SHA-256 before and after reading:

```powershell
python comparison/analyse_saved_predictions.py --predictions "../code/pm25_london_bundle/artifacts/london_3year_final_comparison_20260905_145050/test_2024_laqn_all_models.csv.gz"
```

No original station rows are redistributed here. The saved input filename and hash, rather than a private absolute path, are included in the evidence. The JSON also contains extra descriptive monthly and day-to-day-change summaries; these are not new feature-selection or causal experiments.

## Files and provenance

| File | Purpose |
| --- | --- |
| `REPORT.md` and the `.docx` | Same narrative, generated from one content tree |
| `analyse_saved_predictions.py` | Read-only recomputation of original hybrid errors |
| `build_report.py` | Word report, Markdown and quantitative figure generation |
| `test_report.py` | Seven integrity and report-consistency tests |
| `sources.json` | Primary URLs, paper result locations and comparison caveats |
| `evidence/high_pollution_diagnosis.json` | Aggregate counts, errors, episodes and input hash |
| `evidence/build_audit.json` | Hash checks for the inputs used to build the report |
| `figures/error_concentration.png` | Two 100% bars from the recomputed counts/errors |
| `render_with_word.ps1` | Windows read-only Word-to-PDF-to-PNG QA fallback |

Existing sources are `results/metrics.csv`, `results/provenance.json`, `docs/RESEARCH_PROTOCOL.md`, and `results/outlier_sensitivity_20260909/`. Their roles and limitations are stated in the report. The original experiment commits are `b4a6f6c` and `ea3efdf`.

All report writes resolve inside `comparison`; rendering previews are under the Git-ignored `artifacts/comparison_20260912` directory. Never commit raw data, credentials or checkpoints. No model-selection thresholds were tuned against 2024 for this report.

For Windows visual QA, the usual document renderer was unavailable because LibreOffice was absent. `render_with_word.ps1` uses a newly created hidden Word instance, opens only this report read-only, checks it is unchanged and rasterises it with Poppler. Pass the actual Poppler directory through `-PopplerDirectory`. It does not alter the presentation or other documents.
