# Editorial review and claim-evidence map

## September 29 revision plan and evidence boundary

The maintained manuscript is now `Aitken/research_paper/main.tex`. Its starting
point was the September 7 manuscript in `LY_project/research_paper`; that source
and the original London bundle remain untouched. This revision does not train
models or modify the deployed model, dashboard, presentation or official score
table. It excludes the recent unsuccessful peak-sensitivity experiment.

Mini-outline for the revision, in drafting order:

1. Methods: define the five temporal summaries, their existing clipped lag-one
   source, training-only normalization and the 72-channel input order.
2. Protocol: separate the original archived comparison from the later matched
   temporal comparison, which refitted HGB; disclose its validation bias guard.
3. Results: retain the seven original results and add the two later matched
   results; distinguish numerical-best performance from model promotion.
4. Abstract/conclusion: report RMSE 3.7679, MAE 2.3361 and R2 0.5201 as the
   best recorded experimental result, with its small matched gain and limits.
5. Provenance: regenerate original metrics from saved predictions, transcribe
   temporal aggregates from the versioned CSV/audit/report, and record hashes.

New paragraph roles: temporal motivation -> exact construction -> experiment
contract; matched numerical result -> uncertainty -> validation qualification;
conclusion -> bounded implication -> proposed next evaluation.

| Added claim | Evidence | Verification scope |
|---|---|---|
| Temporal hybrid: MAE 2.336123, RMSE 3.767902, R2 0.520059 | `../results/temporal_model_20260912/metrics.csv` | Checked against recorded aggregate values, not newly recomputed from unavailable temporal predictions. |
| Matched control: RMSE 3.781493; both score 9,619 rows | Same CSV and `audit.json` | Prior audit reports identical rows, unchanged inputs and six checkpoint hashes. |
| Five history channels plus 66 original predictors and HGB | `../experiments/temporal_model/temporal_features.py`, `run_experiment.py` | Read implemented trailing windows, normalization and input composition. |
| Gain about 0.36%; seven-day-block interval [-0.0203,-0.0075] | CSV arithmetic and `REPORT.md` | Difference recalculated; interval transcribed, not rerun. |
| Temporal model is experimental, not official | `REPORT.md`, 2023 screen | Bias below 15 increased from +0.3892 to +0.4024, failing the promotion guard. |

Five-dimension revision self-review:

- Contribution: an incremental temporal-feature result, not architectural novelty.
- Clarity: original hybrid, fresh matched control and temporal hybrid are named separately.
- Empirical strength: small retrospective gain; no claim of prospective superiority.
- Completeness: matched scores and promotion limitation retained; temporal spatial
  correlations or per-station figures are not fabricated from aggregate metrics.
- Soundness: history is past-only locally; reanalysis/composite availability and
  in-sample residual-training limitations still apply.

The remainder records the original manuscript review; its original-run numerical
claims remain valid. The revision above supersedes statements about repository
availability and the absence of an additional temporal comparison.

This is an editorial companion, not additional experimental evidence. It
implements the requested research-paper-writing skill's outline, paragraph
role, claim-support and five-dimension review workflow. The evidence root below
is `../../code/pm25_london_bundle/` relative to the maintained paper directory.

## Mini-outline and paragraph roles

1. **Introduction:** monitoring gap (opening); incremental spatial value
   (question); implementation case study (contribution); retrospective scope
   (limitation).
2. **Related work:** multi-source pollution estimation (context); boosting
   (baseline rationale); U-Net and weighting components (attribution).
3. **Data:** domain and observation units (definition); QC and mixed-cell
   formula (design); covariates, missingness and history (construction and limits).
4. **Methods:** common feature/target contract (overview); tabular blends and
   ANN (comparison design); hybrid equations and loss (design); direct U-Net
   (controlled description, with acknowledged recipe differences).
5. **Experiments and results:** chronology and observed coverage (protocol);
   metrics and saved uncertainty analysis (evidence); common scores, seed
   sensitivity and high-concentration errors (findings).
6. **Discussion and conclusion:** bounded interpretation; temporal, spatial,
   calibration, stacking and statistical limitations; future work distinguished
   from completed experiments.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Grid is 48 x 64 at 1 km; 1,719 domain cells | `data/processed/london_1km_daily/metadata.json`, `quality_report.json` | Supported by saved metadata/quality report. |
| Prepared package contains 444,202 station-days and 294,364 observed cell-days | Same quality report | Supported; two observation units are distinguished. |
| Final fitting uses 203,497 observed cell-days from 2021-2023 | `artifacts/london_3year_unet_refit_20260905_142452/dry_run_protocol.json`; tree training script | Supported. |
| Mixed-source cell target is (4 LAQN + Breathe)/5 | `scripts/prepare_training_data.py`, `build_targets` | Supported by source code; source medians and training weight are separately described. |
| Final source weights are 1 and 0.15 | Final tree, standalone and hybrid training code | Supported; not confused with prepared Breathe weight 0.35. |
| There are 66 retained predictor channels and a 67th HGB channel for the hybrid | Prepared metadata and final trainer channel selection | Supported; generator verifies count and writes ordered appendix. |
| No same-day measured PM2.5 is supplied as a lag predictor | `build_causal_pm_history`; stored causal-lag checks | Supported for local predictor construction; no blanket upstream calibration independence claim. |
| The hybrid combines baseline and scaled residual in transformed space | `scripts/train_london_unet_3year_final.py`, ensemble inference | Supported; equation does not incorrectly add concentration-scale residuals. |
| The standalone uses no HGB map or residual target | `artifacts/london_standalone_unet_20260906/training_source_frozen.py`, `protocol.json`, audit | Supported by saved source/protocol. |
| 9,619 LAQN station-days on 349 dates, ending 2024-12-14 | Extended station prediction table | Independently checked during paper generation. |
| Main error metrics, bias, spatial correlation and high-event MAE | Same table and `evidence/recomputed_metrics.csv` | Independently recomputed, agreement with saved metrics below 1e-6. |
| HGB-hybrid RMSE difference interval includes zero | `artifacts/london_3year_final_comparison_20260905_145050/metrics.json`, `AUDIT.md` | Supported by saved 50,000-repeat bootstrap; not rerun here. |
| Standalone paired differences have positive descriptive intervals | Standalone report and `metrics.json` | Supported by saved analysis; descriptive only. |
| 2024 was previously inspected and is not a newly untouched test | Current handoff, preserved primary manifest and later audit | Explicitly disclosed. |
| Residual-training HGB maps are in-sample | Final tree refit and residual dataset code, audit | Explicitly disclosed; out-of-fold maps are future work. |
| Breathe provider describes reference-network correction | Verified official Breathe London Communities About page | Supported; upstream timing has not been independently audited. |
| Feature smoothing or loss design causes peak underprediction | No isolating final experiment | Not claimed as established; hypotheses only. |
| Standalone architecture is inherently worse, or hybrid is universally best | No such evidence | Removed; results refer to tested configurations. |
| First/novel/state-of-the-art London estimator | Selective context does not establish priority | Not claimed. |
| Accuracy in all grid cells, independent spatial transfer or operational forecasting | No complete spatial truth/blocked prospective protocol | Not claimed. |

## Five-dimension self-review

### 1. Contribution

- **What knowledge is added?** A traceable empirical comparison showing a small
  hybrid gain, weaker tested standalone RMSE and persistent high-event errors.
  **Pass for an implementation case study.**
- **Is the question meaningful?** It tests the incremental value of spatial
  neural processing above existing spatial/history features. **Pass.**
- **Is a non-obvious new method demonstrated?** No. **Needs new research for a
  method-novelty claim; that claim is excluded.**
- **Is the gain surprising or substantial?** The gain over HGB is approximately
  0.41% in RMSE and not clearly separated from zero. **Empirical limitation
  retained; no inflated contribution wording.**
- **Is originality established against all prior work?** No systematic
  literature comparison establishes priority. **Needs additional literature
  assessment for a novelty-driven venue.**

### 2. Writing clarity

- **Can the implemented procedure be understood?** Target aggregation, source
  weights, transforms, blend, architecture and chronology are explicit.
  **Pass.**
- **Are key technical details reported?** The ordered feature schema and
  artifact paths supplement the methods. Exact stored tree bundles and source
  scripts remain necessary for full refitting. **Pass with availability limit.**
- **Is module motivation clear?** Spatial context and baseline correction are
  presented as motivations, without turning them into proven causal benefits.
  **Pass.**
- **Is terminology stable?** Hybrid, standalone, station-day, cell-day,
  retrospective estimation and measured/estimated values remain distinct.
  **Pass.**
- **Does each paragraph support its section?** Reverse outline above matches
  the argument from scope to protocol to findings and limitations. **Pass.**

### 3. Experimental strength

- **Is improvement over HGB meaningful?** Numerically small; statistical and
  practical superiority are unproven. **Needs stronger evidence for a
  superiority claim; current wording is bounded.**
- **Is absolute performance competitive for a chosen venue?** No venue was
  supplied and there is no common-row comparison to external published systems.
  **Unresolved publication judgement, not asserted.**
- **Are gains validated across settings?** One region and one previously
  inspected benchmark year. **Needs new experiments for broad generalisation.**
- **Are failures included?** Negative bias, high-event errors and weaker
  standalone results are reported. **Pass.**

### 4. Evaluation completeness

- **Are all components ablated?** No. Standalone/hybrid and constrained HGB
  are configuration comparisons, not comprehensive causal ablations.
  **Needs new experiments; disclosed.**
- **Are baselines fair and complete?** Common fitting years and evaluation
  rows are used, but tuning budgets and stopping methods differ. External
  baselines and a newly refitted persistence/climatology table are not added
  from incompatible old experiments. **Bounded comparison; disclosed.**
- **Are metrics appropriate?** MAE, RMSE, R2, bias, centred spatial correlation
  and high-event diagnostics are defined. **Pass.**
- **Is evaluation coverage complete?** 349 labelled days rather than 366,
  and station locations rather than full-grid ground truth. **Coverage
  limitation retained.**
- **Are protocols clear?** Selection, final refit, seed roles, benchmark reuse
  and descriptive bootstrap scope are separate. **Pass.**

### 5. Method design soundness

- **Is this a real operational forecast?** No; covariate availability makes it
  retrospective. **Issue-time evaluation is future work.**
- **Are hidden design limitations disclosed?** In-sample stacking, float16
  clipping, source dependence and same-site history are explicit. **Pass for
  transparency; new experiments are needed to repair these limitations.**
- **Is robustness established?** Seed results are reported but broad robustness
  is not established. **Needs independent time/site evaluation.**
- **Do gains justify costs?** No measured cost/latency comparison exists.
  **Not claimed.**
- **Could a reviewer judge the net value insufficient?** Yes, especially for a
  novelty-focused venue. The paper accurately reports the completed work
  without promising acceptance. **Residual publication risk acknowledged.**

## Final truthfulness check

- No invented measurements, targets at unobserved cells, experiments, p-values,
  collaborators, affiliations, grant identifiers, ethics approvals or repository.
- Real external references support only their associated background/source
  claims; they do not certify the local numerical results.
- Derived/imputed predictors and model estimates are clearly labelled.
- The old two-year primary results are not silently mixed into the new table.
- The paper does not describe R2 as percentage accuracy or 25 micrograms/m3
  as an official health threshold.
- The manuscript is suitable for supervisor review as a factual draft; the
  limitations above are not silently treated as completed experiments.
