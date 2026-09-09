# Dashboard implementation and handover

Implemented on 9 September 2026 inside `Aitken/dashboard`. The separate
`code/pm25_london_bundle` was read but not modified. **Aitken now contains the
dashboard and research evidence, not the complete acquisition/training pipeline.**

## Architecture and entry points

| File | Responsibility |
|---|---|
| `dashboard/app/page.tsx` | Explorer, benchmark, request state, SVG map, PNG export |
| `dashboard/app/globals.css` | Responsive layout, design tokens, keyboard/reduced-motion rules |
| `dashboard/lib/research.ts` | Date validation, binary decoding, scales, colour interpolation, grid outline |
| `dashboard/lib/webmcp.ts` | Optional browser-agent tools using the same app actions |
| `dashboard/public/data/` | Curated derived archive, aggregate metrics and provenance |
| `dashboard/tests/research.test.mjs` | Calendar, decoding, geometry, every map chunk and all-model checks |
| `scripts/export_dashboard_data.py` | Reproducible exporter from existing local model artifacts |
| `scripts/stage_dashboard.mjs` | Copies only compiled public assets into the host's static output |

React runs in the browser. vinext produces static HTML, JavaScript and CSS.
There is no inference server, database, external basemap or background training.
The browser fetches three small JSON indexes and then the selected model/month
binary, caching previously used months in memory. A new request cancels the old
request; stale responses cannot replace the current map. Truncated, non-finite
or invalid arrays are rejected. SHA-256 is also checked where Web Crypto is
available. Missing dates and unsupported map models fail explicitly.

## Provenance and data contract

The complete source filenames and SHA-256 hashes are in
`dashboard/public/data/provenance.json`. Important canonical runs:

- `london_3year_tree_refit_20260905_141738`: both HGB variants; target transform.
- `london_3year_unet_refit_20260905_142452`: residual U-Net ensemble maps.
- `london_standalone_unet_20260906`: standalone maps, all-model LAQN comparison
  rows and canonical aggregate scores, including the pre-specified seed 42.

The original station comparison CSV is used locally to verify aggregate scores
and compute daily scores. **Individual observations and station identities are
not included in the dashboard.** Raw satellite/weather/fire files, credentials,
model weights and the original training tensors remain outside GitHub.

| Export | Contents |
|---|---|
| `catalog.json` | Seven models; exact saved aggregate scores; supported dates; chunk URLs, byte lengths and hashes |
| `geometry.json` | 48 rows × 64 columns; 1,719 London indices; EPSG:27700; 1 km transform |
| `daily.json` | Per-day observation count, per-model daily errors and grid summary statistics |
| `grids/{model}-2024-{month}.bin` | Little-endian float32 values, day-major then `geometry.indices` order |

The spatial origin is easting 500,000 m, northing 202,000 m; rows increase
southwards. Only valid London cells are exported; outside-mask hybrid NaNs are
not errors. Normalised log predictions are inverse-transformed using the saved
run parameters, not a newly fitted scaler. The original hybrid grid is stored
at float16 precision; daily scores come from the canonical station predictions,
not the rounded image grid. Other source arrays are exported as float32. Binary
round-trip equality means equality to this exported float32 representation.

All 366 dates of leap-year 2024 have maps for five variants, totalling 1,830 maps
and 60 monthly chunks. The 9,619 LAQN station-days span 349 dates through
14 December. The last 17 dates have predictions but no measured-label error.
ANN and XGBoost remain scores-only because full-grid artifacts were not available;
the UI never substitutes another model's map.

## Visual design and interpretation

The emil-design-eng guidance shaped a restrained, map-first observatory: clear
hierarchy, usable controls, tabular numeric values, modest pointer feedback and
no animations on scientific values. Keyboard navigation and reduced-motion
preferences disable motion. The layout adapts from a map/summary split to a
single column; the results table scrolls horizontally on narrow screens.

Colours use a sequential Viridis interpolation, with a visible legend and
unchanged cell values. Day-specific ranges reveal local variation; the optional
fixed 0–50 µg/m³ range supports comparisons. Values above that fixed maximum
retain their real values but saturate in colour, with a warning. Relative
hotspots use each map's 90th percentile and are off by default. No smoothing,
invented street detail or health-category labelling is applied.

PNG export includes the selected model/date, physical units, scale, grid
resolution and retrospective-archive caveat. The map is a predicted surface,
not a measured or interpolated ground-truth surface.

## Scientific boundaries

The training period is 2021–2023. The 2024 comparison is retrospective and was
inspected in previous experiments. Same-day environmental products and quarterly
satellite composites prevent a strict future-forecast claim. Prior-day PM2.5
features remain historical. The hybrid corrects HGB in transformed target space;
it is not an arithmetic sum of two raw-concentration maps. Standalone U-Nets
use the 66 predictor channels without HGB input.

The hybrid's RMSE is 3.7765 versus HGB's 3.7920 µg/m³. The small difference is
not decisive under the existing descriptive day-bootstrap analysis. R² is not
percentage accuracy. Monitoring-site scores cannot establish performance at
every London cell. Research-only information must not be presented as official
air-quality or health advice.

## Checks and maintenance

`npm test` checks leap days, unsupported dates, corrupt binary buffers, colour
scale boundaries, geometry, all 60 hashes, all 1,830 map ranges/medians, all seven
published metric rows and missing-label handling. TypeScript checks and lint
cover the authored UI/helpers. The static production build is also checked.
Keep the exact lockfile, and use `npm ci` when installing elsewhere.

The optional WebMCP tools are `read_model_benchmark` and
`generate_archived_heatmap`. They feature-detect `document.modelContext`,
register with lifecycle cleanup and validate inputs before invoking the same
state/actions as the visible UI. No model or external account is altered.
Broad browser visual QA and a downloaded-PNG inspection are not claimed by the
automated data/build checks; manually inspect the display before a formal talk.

Validation on 9 September 2026: all seven automated test groups, TypeScript and
authored-code lint passed. All 1,830 maps passed range/median checks and all 60
monthly files passed SHA-256 checks. A production build completed successfully.
In the supported browser agent-tool interface, both tools registered with their
expected schemas; reading the benchmark returned all seven correct scores,
generating standalone U-Net for 29 February and hybrid for 15 July returned
`displayed` with 1,719 cells and the expected map statistics. An extra input field
to the read tool and an impossible 30 February date were intentionally rejected.
The final map was restored to the default hybrid example. No broad screenshot
or click-through visual review is implied by these focused contract checks.

The starter's vulnerable packages were patched. `sharp` is pinned through an
override to 0.35.4 for the development toolchain; remove that override only when
the upstream dependency resolves to an equally patched version. Never use
`npm audit fix --force` blindly.

## Extending the project

For more historical dates, first create and verify the required input features
and frozen-model predictions in the research pipeline, then update the export,
catalog and calendar checks. For genuinely new-day inference, add a separately
validated backend and input-availability contract. Do not relabel archive
rendering as live inference. To add ANN/XGBoost maps, export their actual trained
model predictions on the same 1 km grid and add provenance before enabling them.
