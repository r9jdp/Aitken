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
        lines += ['This is a partial/failed run; no overall success claim is justified.', '']
    for family in ['standalone','hybrid']:
        selected=summary['selection'].get(family,{}).get('selected')
        lines.append(f"- **{family.title()} U-Net:** " + (f"{selected} passed the 2023 screening gate; see the matched 2024 confirmation below." if selected else 'no capping treatment passed all pre-specified 2023 criteria. Keep the existing model.'))
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
        lines += ['No new 2024 evaluation was performed because no capped recipe passed the validation gate. '
                  'The existing published 2024 scores are unchanged. Do not substitute the 2023 screening numbers into the presentation’s 2024 results table.', '']
    final_rows=[]
    for family, results in summary['final'].items():
        lines += [f'### {family.title()}—three-seed ensemble', '', '| Treatment | MAE | RMSE | R² | Bias | MAE ≥25 |', '|---|---:|---:|---:|---:|---:|']
        for name,m in results.items():
            if name=='paired_block_interval': continue
            lines.append(f"| {name} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | {m['bias']:.4f} | {m['high']['mae']:.4f} |")
            final_rows.append({'model':family,'treatment':name,**{k:m[k] for k in ['n','mae','rmse','r2','bias']},'high_mae':m['high']['mae']})
        ci=results['paired_block_interval']
        lines += ['', f"Candidate minus control RMSE: {ci['difference']:+.4f} µg/m³; paired seven-day-block 95% interval [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]. An interval containing zero does not establish a reliable improvement.", '']
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
              '- The 2024 benchmark was already examined previously; any confirmation there is retrospective. No performance guarantee or universal superiority claim is made.',
              '', '[Experiment protocol and commands](../../experiments/outlier_sensitivity/README.md) · '
              '[Source provenance](../../experiments/outlier_sensitivity/PROVENANCE.md)', '',
              'Winsorisation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html', '']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(output/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    publish(a.run_dir,a.output_dir)
