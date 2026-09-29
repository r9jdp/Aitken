# Greater London PM2.5 manuscript

This is the maintained LaTeX paper inside Aitken, updated on **29 September
2026** from the original `LY_project/research_paper` draft. The original paper
and London bundle were not edited. Author and supervisor attribution are
preserved: Rajdeep Pandey, KJ Somaiya College of Engineering; guidance from
Dr. Suchitra Patil.

## What changed

- Abstract, method, comparison table, RMSE figure and conclusion now include
  the temporal hybrid: MAE **2.3361**, RMSE **3.7679**, R2 **0.5201**.
- The fresh matched control (RMSE **3.7815**) is shown separately from the
  original official hybrid (RMSE **3.7765**).
- The five past-only features, 72-input composition, small 0.36% matched RMSE
  gain, saved seven-day-block interval, and validation qualification are explicit.
- Existing baseline/standalone results and scientific limitations remain.
- The recent unsuccessful peak-sensitivity test is not described in the paper.
- This is a manuscript update, not model promotion or retraining. The official
  dashboard, performance table and presentation are unchanged.

The temporal model is the best recorded numerical configuration but remains
experimental: it missed the 2023 normal-range positive-bias guard. 2024 is a
previously examined **retrospective benchmark**, not an untouched test set.

## Build / Overleaf

Upload `london_pm25_latex.zip` into Overleaf, with `main.tex` as the main file.
The ZIP includes source, bibliography, table fragments and PDF figures; no
training data or API access is needed for typesetting. Alternatively, from this
directory use a locally installed compiler:

```powershell
tectonic --untrusted --keep-logs main.tex
```

Or run pdfLaTeX, BibTeX, and pdfLaTeX twice. The compiled `main.pdf`, ZIP and
temporary previews remain local/Git-ignored. The source and small figure PDFs
needed to compile it are tracked. Check `evidence/validation.json` for the
actual verification status of this revision; opening the source alone is not
proof of compilation.

## Reproduce the evidence-backed assets

From the Aitken repository root:

```powershell
py -3.10 research_paper/build_assets.py --temporal-only
py -3.10 research_paper/build_assets.py --bundle ../code/pm25_london_bundle
```

The first command only reads the small versioned temporal result files. The
second additionally requires the existing original prediction/metadata
artifacts, NumPy, pandas and Matplotlib. Source inputs are read-only; all
generated outputs remain in this manuscript directory. No models are trained.

Original scores are recomputed from saved station predictions and checked
against saved metrics to within 1e-6. Temporal scores are checked against
`../results/temporal_model_20260912/metrics.csv`, `audit.json` and `REPORT.md`.
The prior audit reports identical evaluation rows and six checkpoint hashes,
but the temporal prediction arrays/checkpoints are unavailable locally for
this update. No fresh prediction-level temporal audit is claimed. Both
bootstrap analyses are transcribed from saved evidence, not newly rerun.

## Contents

- `main.tex`, `references.bib`: paper and 18 references, including an explicitly
  labelled project-evidence citation, not an additional peer-reviewed baseline.
- `tables/`, `figures/`: included small assets for independent LaTeX compilation.
- `evidence/`: original recomputed metrics and both provenance manifests.
- `REVIEW_AND_CLAIMS.md`: outline, claim-evidence mapping and editorial review.
- `build_assets.py`: asset regeneration from existing artifacts.

No raw data, model weights, credentials or synthetic observations are included.
The manuscript does not establish general U-Net superiority, operational
forecast skill, accuracy at every unmonitored cell, or state-of-the-art status.
