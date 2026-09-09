# Aitken dashboard

A responsive London PM2.5 observatory built with React, TypeScript, vinext/Vite
and shadcn/Base UI. No training, database, external tile service or API key is
needed to use it. The dashboard does not depend on the original London folder
at runtime.

## Run locally

Use Node.js **22.18 or newer**. From `Aitken`:

```powershell
npm --prefix dashboard ci
npm --prefix dashboard run dev
```

Open the address printed in the terminal, normally `http://localhost:3000`.
If that port is already occupied, a different port may be chosen. Stop with Ctrl+C.
Do not open the built HTML directly as a `file://` URL; the archive needs HTTP.

## What you can do

- **Heatmap explorer:** choose a model and any day from 1 January to 31 December
  2024, then Generate heatmap. Previous/next buttons load adjacent days directly.
- Switch between a day-specific colour scale and a fixed 0–50 µg/m³ scale.
  Use the fixed scale when comparing different dates or models.
- Optionally outline the highest predicted 10% of cells on that day. These are
  relative hotspots, not a health-limit classification.
- Download a labelled PNG containing the model, date, scale and research caveat.
- **Model benchmark:** compare all seven variants by MAE, RMSE and R².
- Expand the research notes for the evaluation design and limitations.

Five map variants are available: hybrid HGB + residual U-Net, unconstrained HGB,
limited-monotonic HGB, standalone three-seed U-Net and standalone seed-42 U-Net.
ANN and XGBoost have evaluation results but no full-grid archive in this release.

## Important scientific scope

These are **saved model predictions**, rendered on demand; not newly inferred
future dates. The models were fitted on 2021–2023. The benchmark uses the same
9,619 LAQN station-days over 349 dates in 2024. On 15–31 December there are maps
but no evaluation labels; the UI deliberately shows no daily error then.

No model was retrained for the dashboard. Neither colours nor optional hotspot
outlines change prediction values. Monitoring-site accuracy does not validate
every unmonitored cell. The hybrid's small numerical lead over HGB is not a
decisive advantage. More in [the handover](../docs/DASHBOARD.md).

## Check and build

From `Aitken`:

```powershell
npm --prefix dashboard run typecheck
npm --prefix dashboard run lint
npm --prefix dashboard test
npm --prefix dashboard run build
node scripts/stage_dashboard.mjs
```

`dashboard/dist/client` is the static production build; the last command stages
its public contents into `Aitken/dist` for the configured host. Build directories
and `node_modules` are ignored by Git. The lockfile is committed. The linter
checks authored application/helper/test code, not the unchanged bundled shadcn
component library.

For a production-like local preview, from `Aitken/dashboard`, run `npm start`
after building. A static HTTP server serving `dashboard/dist/client` also works.

## Refreshing derived outputs

Only needed when the original research artifacts change. Requires Python with
NumPy and pandas and the separate complete London bundle:

```powershell
py -3.10 scripts/export_dashboard_data.py --bundle ../code/pm25_london_bundle
```

Run this from `Aitken`, then rerun all dashboard checks. The exporter checks
canonical station-row metrics, restores target units where necessary, exports
only London cells and verifies the binary round trip. It never trains models.

Do not add raw observations, downloaded environmental products, credentials or
model checkpoints to `public/`. Everything under `public/` is downloadable by a
dashboard visitor. The current derived archive is about **12.6 MB** of grids,
split into 60 monthly files (~0.2 MB each) loaded only as needed.
