# Report review and claim evidence map

Reviewed on 12 September 2026. This is a project-comparison and diagnostic report, not a new training study or an exhaustive literature review.

## Outline and paragraph roles

1. **Opening and verdict:** retain the original hybrid, with a small and uncertain advantage over HGB.
2. **Method and scope:** distinguish standalone from residual U-Net, explain the data and retrospective evaluation.
3. **External evidence:** compare relevant London studies and recent U-Net work without a false head-to-head ranking.
4. **Challenge and evidence:** quantify high-concentration errors and show their disproportionate contribution.
5. **Diagnosis and limitations:** give episode examples, distinguish training/input hypotheses from demonstrated atmospheric causes.
6. **Completed experiment:** report the failed capping intervention with matched controls and uncertainty.
7. **Recommendations and interpretation:** state future tests, defensible faculty wording and reproducibility; conclude with linked references.

## Five dimension self review

| Dimension | Question | Answer and resulting safeguard |
| --- | --- | --- |
| Contribution | Does the report claim an algorithmic breakthrough? | No. It documents an implemented London product and a reproducible error diagnosis. “Best” is limited to the numerical release comparison. |
| Writing clarity | Can a group member explain the main result? | The report defines HGB, ANN, MAE, RMSE, CV, AOD and PLS, gives a simple squared-error example and includes a faculty-ready paragraph. Technical details are retained only where they affect interpretation. |
| Experimental strength | Are the comparisons matched, and what is actually new? | All seven release rows share 9,619 observations. The capping test has separate matched fresh controls; no models were trained for this report. Literature results are not a matched experiment. |
| Evaluation completeness | Are negative results, peaks and missing scope exposed? | Yes: all 216 high observations are retained, episode and ordinary-day errors are reported, capping failure and its interval are included, and 2024 is labelled retrospective. All-cell, new-site and real-time forecast accuracy remain unestablished. |
| Method design soundness | Are causal explanations or data faults being assumed? | No. The report identifies possible objective/input mismatches but calls for controlled tests. It notes in-sample baseline maps and recommends out-of-fold residual training. High PM2.5 is not equated with noise. |

## Claim evidence map

| Major claim | Evidence | Status |
| --- | --- | --- |
| Original hybrid MAE 2.3405, RMSE 3.7765 and R² 0.5179 | `results/metrics.csv`; original saved predictions recomputed in `evidence/high_pollution_diagnosis.json` | Supported |
| The hybrid's lead over HGB is small and not decisive | `docs/RESEARCH_PROTOCOL.md`, saved 50,000-repeat paired calendar-day interval including zero | Supported, descriptive/post-selection |
| 216 high observations are 2.25% of rows but 38.76% of squared error | Recomputed subgroup count, sum of squared residuals and total from all 9,619 saved predictions | Supported |
| Every high observation is underpredicted | `at_least_25.underprediction_count == 216` | Supported for this model and benchmark |
| 11 March accounts for 13.68% of total squared error | `highest_error_days` aggregate; 33 matched evaluated sites | Supported; date selected descriptively by error |
| Log compression and SmoothL1 may contribute to peak underprediction | Fixed training recipe and frozen HGB log-blend weight; loss functions emphasise errors differently from raw squared error | Mechanism is plausible; causal contribution needs evidence |
| Regional atmospheric information could help | Different predictors used in the cited London/GB studies | Hypothesis; needs matched input ablation |
| Training capping did not improve the final standalone result | `results/outlier_sensitivity_20260909/results.json`, 2024 matched control vs cap; positive seven-day-block ΔRMSE interval | Supported for the tested treatment/ensembles |
| Neither hybrid cap passed development criteria | Same source, 2023 screening; MAE and high-MAE gate | Supported; no 2024 hybrid cap rerun claimed |
| Published models are contextual comparators, not a leaderboard | Different study years, target construction, inputs, spatial coverage, temporal resolution and validation schemes | Supported; no common-benchmark superiority claim |
| The project identifies measurement noise or atmospheric causes | No sensor-fault ground truth or episode-attribution experiment | Not supported; explicitly not claimed |

## Source caveats preserved

- **Danesh Yazdi 2020:** daily 1 km London is the strongest task match. Monitor-held-out cross-validation and augmented labels differ from our 2024 temporal benchmark. Overall and spatial R² are not interchangeable.
- **Dimakopoulou 2022:** 0.66–0.83 describes different models. The study's hybrid models are not HGB plus U-Net. A questionable low RMSE entry in the published table is not imported into a superiority claim.
- **Schneider 2020:** the aggregate score is Great Britain-wide, not London-only. Its regression-based validation statistics should not be treated as identical to our implementation's R² calculation.
- **Legaria-Santiago 2026:** the quoted values are specifically Scenario 2 at Marylebone and Camden, with traffic and neighbouring pollution. The journal reference is confirmed; the linked author manuscript exposes the result table.
- **Galindo-Prieto 2026:** the quoted PLS results are on the original scale at London site CT3, not a city grid. The paper includes imputation and reports both transformed and original-scale metrics. Inconsistent values in other model rows are not copied.
- **Sharma 2025:** publisher metadata/abstract verify the Oslo dispersion-surrogate task. No inaccessible numerical result has been invented.
- **Ronneberger 2015:** architecture reference only, not a London PM2.5 comparison.
- Primary source URLs and result-table locators are recorded in `sources.json`. A publisher-reported number is not an independently reproduced result.

## Verification record

- Original prediction file hash: `e8de3610cf9da8e970262b18910135a969bd7f5fcfda290be6ffe95097227bc3`.
- Aggregate recomputation preserves all evaluation observations and reproduces the original hybrid release metrics to numerical tolerance.
- The raw prediction gzip uses a byte-exact SHA-256. The separate build audit normalises CRLF to LF for tracked text inputs, so Git line-ending conversion does not falsely look like a changed result on a fresh checkout.
- Seven report tests pass: release-table agreement, subgroup conservation, cap-screen agreement, matched-final agreement, unchanged input hashes, output containment and citation/document structure.
- Word and Markdown use one shared content tree; the figure is drawn from the recomputed aggregate data.
- Final Word preview: **8 US Letter portrait pages**, rendered with Microsoft Word 2021 read-only and bundled Poppler. Each page was visually inspected for table alignment, text clipping, citation spacing and completeness.
- The first layout pass revealed an inherited decorative Title rule and citation spacing; both were corrected. Final headings are black, tables have consistent widths and padding, and all ten numbered sources have external hyperlink relationships.
- The canonical `render_docx.py` could not run because this Windows runtime has no LibreOffice. The initial sandboxed Word instance also stalled; only that task-created hidden automation instance was stopped. A permitted hidden Word export then completed, and its source-DOCX hash was unchanged.
- QA previews remain Git-ignored under `artifacts/comparison_20260912`; the committed deliverables are the Word document, Markdown, aggregate figure/evidence and reproduction/review files inside `comparison`.
- No training, model activation, dashboard change, presentation edit or change to the old London bundle was performed for this report.
