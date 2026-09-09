"""Render a shareable aggregate report; never modifies the dashboard/model table."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

# Restrict plotting caches as well as report outputs to Aitken.
ROOT = Path(__file__).resolve().parents[2]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'artifacts' / 'outlier_plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_experiment import inside_repo, json_save


def publish(run, output):
    run, output = inside_repo(run), inside_repo(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run/'summary.json').read_text())
    rows = []
    for family, treatments in summary['screening'].items():
        for treatment, value in treatments.items():
            m, c = value['metrics'], value['cap']
            rows.append({'model':family, 'treatment':treatment, 'cap_pm25':c['cap_pm25'],
                         'affected_training_cell_days':c['affected_cell_days'],
                         **{k:m[k] for k in ['n','mae','rmse','r2','bias']},
                         'high_n':m['high']['n'], 'high_mae':m['high']['mae'],
                         'high_rmse':m['high']['rmse'], 'high_bias':m['high']['bias'],
                         'below25_mae':m['below25']['mae'], 'high_recall':m['high_recall'],
                         'high_precision':m['high_precision']})
    pd.DataFrame(rows).to_csv(output/'screening_metrics.csv',index=False)
    # The curated JSON has no credentials, absolute input paths, or station rows.
    json_save(output/'results.json', summary)
    lines=['# Does capping extreme training values help the London U-Nets?', '',
           '## Conclusion', '', f"Experiment status: **{summary['status']}**.", '']
    if summary['status'] != 'complete':
        lines += ['Work is incomplete; no final success claim is justified.', '']
    for family in ['standalone','hybrid']:
        selected=summary['selection'].get(family,{}).get('selected')
        confirmed=summary['final'].get(family)
        if confirmed:
            m,c=confirmed[selected],confirmed['control']
            delta=m['rmse']-c['rmse']
            guard=m['mae']<=c['mae'] and m['high']['mae']<=c['high']['mae']
            verdict='numerically improved on 2024' if delta<0 and guard else 'did not provide a consistent improvement on 2024; keep the existing model'
            lines.append(f"- **{family.title()} U-Net:** {selected} {verdict}. Matched-control ΔRMSE: {delta:+.4f} µg/m³. See the uncertainty interval below.")
        else:
            lines.append(f"- **{family.title()} U-Net:** " + (f"{selected} passed the 2023 screen, but confirmation is not complete." if selected else 'no capping treatment passed all pre-specified 2023 criteria. Keep the existing model.'))
    lines += ['', 'The caps modify experimental training supervision, not verified measurement errors. '
              'Every validation/test reading—including pollution episodes—remains unchanged. '
              'PM2.5 ≥25 µg/m³ is a diagnostic group, not AQI or an automatic outlier rule.', '',
              '## 2023 screening: train on 2021–2022', '',
              'All comparisons below use the same original LAQN observations, seed 42, fixed schedules and matched fresh initialisation. '
              'Lower MAE/RMSE is better; higher R² is better. Units are µg/m³ except R².', '',
              '| Model | Training treatment | MAE | RMSE | R² | Bias | MAE ≥25 | Targets capped |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['treatment']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['r2']:.4f} | {r['bias']:.4f} | {r['high_mae']:.4f} | {r['affected_training_cell_days']:,} |")
    if rows:
        lines += ['',f"Evaluation: **{rows[0]['n']:,} station-days**, including **{rows[0]['high_n']:,} observations ≥25 µg/m³**. "
                  'Training-only caps are 43.83 µg/m³ for cap_995 and 38.58 µg/m³ for cap_990; readings below those caps are unchanged.']
    lines += ['', '**Pass rule:** RMSE must decrease, while neither overall MAE nor MAE at observed PM2.5 ≥25 may increase. '
              'The lowest-RMSE eligible cap is selected; no thresholds are selected using 2024.', '']
    for family, treatments in summary['screening'].items():
        control=treatments['control']['metrics']
        lines.append(f"### {family.title()}: change versus its matched control\n")
        for treatment, value in treatments.items():
            if treatment=='control': continue
            m=value['metrics']
            lines.append(f"- {treatment}: ΔRMSE {m['rmse']-control['rmse']:+.4f}; ΔMAE {m['mae']-control['mae']:+.4f}; Δhigh-pollution MAE {m['high']['mae']-control['high']['mae']:+.4f} µg/m³.")
        lines.append('')
    lines += ['## 2024 confirmation', '']
    if not summary['final']:
        pending=any(d['selected'] for d in summary['selection'].values())
        reason=('A capped recipe passed screening, but the matched three-seed 2024 confirmation is not yet complete.' if pending else
                'No new 2024 evaluation was performed because no capped recipe passed the validation gate.')
        lines += [reason+' The existing published 2024 scores are unchanged. Do not substitute the 2023 screening numbers into the presentation’s 2024 results table.', '']
    final_rows=[]
    for family, results in summary['final'].items():
        lines += [f'### {family.title()}—three-seed ensemble', '', '| Treatment | MAE | RMSE | R² | Bias | MAE ≥25 |', '|---|---:|---:|---:|---:|---:|']
        for name,m in results.items():
            if name=='paired_block_interval': continue
            lines.append(f"| {name} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | {m['bias']:.4f} | {m['high']['mae']:.4f} |")
            cap=json.loads((run/'final'/family/f'{name}_cap.json').read_text())
            final_rows.append({'model':family,'treatment':name,**{k:m[k] for k in ['n','mae','rmse','r2','bias']},
                               'cap_pm25':cap['cap_pm25'],'affected_training_cell_days':cap['affected_cell_days'],
                               **{'high_'+k:m['high'][k] for k in ['n','mae','rmse','bias']},'below25_mae':m['below25']['mae']})
        ci=results['paired_block_interval']
        ci_interpretation=('This interval is entirely above zero: the capped model has worse RMSE under this resampling analysis.' if ci['ci95'][0]>0 else
                           'This interval is entirely below zero: it supports lower RMSE under this resampling analysis.' if ci['ci95'][1]<0 else
                           'This interval contains zero: it does not establish a reliable improvement.')
        lines += ['', f"Candidate minus control RMSE: {ci['difference']:+.4f} µg/m³; paired seven-day-block 95% interval [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}] ({ci['replicates']:,} resamples). {ci_interpretation}", '']
        directory=run/'final'/family
        chosen=summary['selection'][family]['selected']
        cap=json.loads((directory/f'{chosen}_cap.json').read_text())
        m=results[chosen]
        lines += [f"Evaluation uses **{m['n']:,} unchanged station-days**, including **{m['high']['n']} observations ≥25 µg/m³**. "
                  f"The selected training-only cap is **{cap['cap_pm25']:.4f} µg/m³**, affecting **{cap['affected_cell_days']:,} training cell-days**. "
                  'The three members use seeds 42, 11 and 22; each capped member has a matched freshly trained control.', '',
                  '| Treatment | MAE below 25 | RMSE ≥25 | Bias ≥25 |', '|---|---:|---:|---:|']
        for name,m in results.items():
            if name=='paired_block_interval': continue
            lines.append(f"| {name} | {m['below25']['mae']:.4f} | {m['high']['rmse']:.4f} | {m['high']['bias']:.4f} |")
        lines.append('')
        lines += ['The same pre-specified 2024 dates are shown for both treatments: 15 July (ordinary-day example) and 11 March (previously diagnosed severe episode). These dates were not used to choose the cap.', '',
                  '| Date | Treatment | Stations | Observed mean | Predicted mean | Station RMSE |', '|---|---|---:|---:|---:|---:|']
        for day in ['2024-07-15','2024-03-11']:
            for name in results:
                if name=='paired_block_interval': continue
                df=pd.read_csv(directory/f'{name}_daily.csv')
                for r in df[df.date==day].itertuples():
                    lines.append(f'| {day} | {name} | {r.n} | {r.observed_mean:.3f} | {r.predicted_mean:.3f} | {r.rmse:.3f} |')
        lines.append('')
    if final_rows: pd.DataFrame(final_rows).to_csv(output/'confirmation_metrics.csv',index=False)
    lines += ['## Ordinary days and high-pollution episodes', '',
              'The figure shows all 2023 days, not only selected successes. Observed and predicted lines are daily means across the same evaluated LAQN stations.', '',
              '![2023 daily validation comparison](daily_comparison.png)', '']
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    colors={'control':'#2367aa','cap_995':'#d58619','cap_990':'#ad3d4a'}
    for ax,family in zip(axes,['standalone','hybrid']):
        directory=run/'screening'/family
        for i,treatment in enumerate(summary['screening'].get(family,{})):
            df=pd.read_csv(directory/f'{treatment}_daily.csv')
            dates=pd.to_datetime(df.date)
            if i==0: ax.plot(dates,df.observed_mean,color='#202020',lw=1.3,label='Observed')
            ax.plot(dates,df.predicted_mean,color=colors[treatment],lw=.8,alpha=.9,label=treatment)
        ax.set_title(f'{family.title()} U-Net: unchanged 2023 validation readings',loc='left',fontsize=11)
        ax.set_ylabel('Daily mean PM2.5 (µg/m³)')
        ax.grid(alpha=.18)
        ax.legend(loc='upper right',ncol=4,fontsize=8)
        if (directory/'control_daily.csv').exists():
            control=pd.read_csv(directory/'control_daily.csv')
            peak=str(control.loc[control.observed_mean.idxmax(),'date'])
            lines += [f'### {family.title()} daily examples', '',
                      f'15 July is the fixed ordinary-day example. {peak} is the highest observed daily mean in this development year (descriptive episode selection, not an independent test).', '',
                      '| Date | Treatment | Stations | Observed mean | Predicted mean | Station RMSE |', '|---|---|---:|---:|---:|---:|']
            for day in dict.fromkeys(['2023-07-15',peak]):
                for treatment in summary['screening'][family]:
                    df=pd.read_csv(directory/f'{treatment}_daily.csv')
                    for r in df[df.date==day].itertuples():
                        lines.append(f'| {day} | {treatment} | {r.n} | {r.observed_mean:.3f} | {r.predicted_mean:.3f} | {r.rmse:.3f} |')
            lines.append('')
    fig.savefig(output/'daily_comparison.png',dpi=160)
    plt.close(fig)
    lines += ['## Integrity, limitations and reproducibility', '',
              f"- Original input/reference hashes unchanged: **{summary.get('input_integrity_passed')}**.",
              f"- Validation observations unchanged: **{summary.get('validation_observations_unchanged')}**.",
              f"- Training/inference budget used: {summary.get('gpu_budget_seconds_used',0)/60:.1f} minutes (not total implementation time).",
              '- No HGB, ANN or XGBoost retraining. No new data, feature changes or architecture changes.',
              '- All new experiment files are inside Aitken; original observations, dashboard, published model table and presentation were left unchanged.',
              '- The screen is one seed on previously used 2023 validation data, not independent proof of generalisation. Three-seed confirmation is conditional on passing the gate.',
              '- The fixed-schedule, deterministic matched controls may differ slightly from historic runs. Compare treatments against these controls, not unmatched old numbers.',
              '- The block interval captures variation across calendar-day blocks for these fitted ensembles, not uncertainty across all possible training runs or future cities.',
              '- The 2024 benchmark was already examined previously; any confirmation there is retrospective. No performance guarantee or universal superiority claim is made.',
              '', '[Experiment protocol and commands](../../experiments/outlier_sensitivity/README.md) · '
              '[Source provenance](../../experiments/outlier_sensitivity/PROVENANCE.md)', '',
              '[Independent saved-output audit](audit.json) verifies original labels, checkpoints, matched initialisation and recalculated scores.', '',
              'Method reference: [SciPy winsorisation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html). '
              'This runner uses NumPy linear-interpolated percentiles followed by an upper cap, not a direct call to SciPy’s rank-based function.', '']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    note=['# Short explanation for the presentation', '',
          'We kept the same U-Net architectures and tested one small preprocessing change: '
          'capping only the highest 0.5% or 1% of training targets. This is called winsorisation. '
          'We did not replace all readings above 25 with the median, and we did not change '
          'any actual validation or test observations.', '']
    for family in ['standalone','hybrid']:
        decision=summary['selection'].get(family,{})
        chosen=decision.get('selected')
        if chosen:
            screen=summary['screening'][family]
            confirmation=('We then tested matched three-seed ensembles on the original 2024 observations.' if family in summary['final'] else 'The planned matched three-seed 2024 confirmation is not yet complete.')
            note.append(f"For the {family} U-Net, the selected cap lowered 2023 validation RMSE from {screen['control']['metrics']['rmse']:.4f} to {screen[chosen]['metrics']['rmse']:.4f} µg/m³. {confirmation}")
            if family in summary['final']:
                f=summary['final'][family]
                note.append(f"The 2024 RMSE was {f['control']['rmse']:.4f} without capping and {f[chosen]['rmse']:.4f} with capping. The corresponding R² values were {f['control']['r2']:.4f} and {f[chosen]['r2']:.4f}. High-pollution MAE changed from {f['control']['high']['mae']:.4f} to {f[chosen]['high']['mae']:.4f} µg/m³.")
                ci=f['paired_block_interval']['ci95']
                interpretation=('The interval is above zero, supporting worse RMSE with capping in this retrospective comparison.' if ci[0]>0 else
                                'The interval is below zero, supporting lower RMSE in this retrospective comparison.' if ci[1]<0 else
                                'The interval includes zero, so we cannot claim a reliable improvement.')
                note.append(f"The paired seven-day-block 95% interval for the RMSE change was [{ci[0]:+.4f}, {ci[1]:+.4f}] µg/m³. {interpretation}")
                if f[chosen]['rmse']>=f['control']['rmse'] or f[chosen]['mae']>f['control']['mae'] or f[chosen]['high']['mae']>f['control']['high']['mae']:
                    note.append('Recommendation: do not adopt this cap. The validation improvement did not carry over consistently to 2024; keep the current model.')
        else:
            note.append(f"For the {family} U-Net, neither cap improved RMSE without worsening overall or high-pollution MAE on validation, so we kept the existing model and did not perform an additional 2024 trial for it.")
        note.append('')
    note += ['High pollution is not automatically bad data. This experiment tests whether mild training-target capping helps; it does not prove the extreme observations were wrong. The project concept, dashboard and original presentation results remain unchanged. Any 2024 comparison is retrospective because that year was examined earlier.', '']
    (output/'PRESENTATION_NOTE.md').write_text('\n'.join(note),encoding='utf-8')
    print(output/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    publish(a.run_dir,a.output_dir)
