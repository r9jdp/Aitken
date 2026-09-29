# Greater London PM2.5 manuscript

This is the maintained LaTeX paper inside Aitken, updated on **30 September
2026** from the original `LY_project/research_paper` draft. The original paper
and London bundle were not edited. Author names and roll numbers are copied
from slide 2 of `../code/pm25_london_bundle/presentation.pptx` (relative to
the Aitken root):

| Group member | Roll number |
|---|---|
| Rajdeep Pandey | 16014223064 |
| Sagar Jadhav | 16014223070 |
| Sohom Mallick | 16014223083 |
| Vruddhi Mule | 16014223099 |

The presentation identifies Group 38, Artificial Intelligence & Data Science,
KJ Somaiya School of Engineering, with guidance from Dr. Suchitra Patil.

## What changed

- Abstract, data, methods, evaluation, comparison tables, RMSE figure and
  conclusion now include the reproduced **temporal hybrid + same-day AURN
  correction**: MAE **2.1343**, RMSE **3.2309**, R2 **0.6471**.
- A new daily-mean plot shows both the full labelled period and the March
  episode, including remaining large peak errors. All plotted values come
  from real observations and saved/reproduced predictions.
- The new result is separate from the temporal hybrid (**3.7679** RMSE) and
  does not change the standalone U-Net score or imply architecture superiority.
- The fresh matched control (RMSE **3.7815**) is shown separately from the
  original official hybrid (RMSE **3.7765**).
- Both the five past-only features and four external AURN correction inputs
  are defined. Same-day information, wind-scaling defect, local reproduction
  boundary and previously inspected 2024 evaluation remain explicit.
- Existing baseline/standalone results and scientific limitations remain.
- The recent unsuccessful peak-sensitivity test is not described in the paper.
- This is a manuscript update, not model promotion or retraining. The official
  dashboard, performance table and presentation are unchanged.

The AURN-assisted configuration has the best reproduced numerical score but
remains experimental. The upstream temporal model missed its 2023 positive-bias
guard, and the regional wind feature needs a separately evaluated correction.
2024 is a previously examined **retrospective benchmark**, not an untouched
test set. Extra same-day station information does not establish next-day skill.

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

The first command refreshes the temporal tables and provenance from versioned
results and verifies supplied temporal predictions when available. The second
additionally requires the original prediction/metadata artifacts and local
AURN reproduction outputs, NumPy, pandas and Matplotlib. Source inputs are read-only; all
generated outputs remain in this manuscript directory. No models are trained.

Original scores are recomputed from saved station predictions and checked
against saved metrics to within 1e-6. Temporal scores are checked against
`../results/temporal_model_20260912/metrics.csv`, `audit.json` and `REPORT.md`.
The prior temporal audit reports identical evaluation rows and six checkpoint
hashes. The supplied supplement now contains the temporal arm's 2023/2024
station predictions. The AURN audit verifies these against the original rows,
repeats correction selection/fitting, and independently recomputes the new
seven-day interval. The upstream neural predictions are not regenerated; the
old non-temporal matched predictions and neural checkpoints are still absent.
The older original/temporal comparison intervals remain transcribed evidence.

Reproduction code and audit: `scripts/reproduce_aurn_supplement.py` and
`comparison/aurn_import_20260930/`, relative to Aitken. Manuscript tables and
figures use the separate local rerun under `artifacts/aurn_reproduction_20260930/`.
The full asset build requires those locally installed artifacts. Neither the
manuscript nor this asset build changes any model or original input data.

## Contents

- `main.tex`, `references.bib`: paper and 20 references, including two explicitly
  labelled project-evidence citations, not additional peer-reviewed baselines.
- `tables/`, `figures/`: included small assets for independent LaTeX compilation.
- `evidence/`: original recomputed metrics, temporal/AURN provenance and author source.
- `REVIEW_AND_CLAIMS.md`: outline, claim-evidence mapping and editorial review.
- `build_assets.py`: asset regeneration from existing artifacts.

No raw data, model weights, credentials or synthetic observations are included.
The manuscript does not establish general U-Net superiority, operational
forecast skill, accuracy at every unmonitored cell, or state-of-the-art status.
