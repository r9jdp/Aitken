"""Recompute paper tables and draw figures from existing, immutable predictions.

Run: py -3.10 research_paper/build_assets.py
No model is trained, no observations are generated, and no source is modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent.parent / 'code' / 'pm25_london_bundle'
ART = BUNDLE / 'artifacts'
SOURCE = ART / 'london_standalone_unet_20260906/test_2024_laqn_all_models.csv.gz'
FINAL = ART / 'london_3year_final_comparison_20260905_145050/metrics.json'
MODELS = {
    'Residual U-Net (3-seed)': 'Hybrid HGB--U-Net (3 seeds)',
    'HGB (unconstrained)': 'HGB (unconstrained)',
    'HGB (limited monotonic)': 'HGB (limited monotonic)',
    'XGBoost': 'XGBoost',
    'ANN (exact sklearn, 5-seed)': 'ANN (5 seeds)',
    'Standalone U-Net (3-seed)': 'Standalone U-Net (3 seeds)',
    'Standalone U-Net (seed 42)': 'Standalone U-Net (seed 42)',
}
TEMPORAL = ROOT.parent / 'results' / 'temporal_model_20260912'


def temporal_assets():
    """Transcribe audited aggregates; do not claim a new prediction-level audit."""
    metrics_path = TEMPORAL / 'metrics.csv'
    audit_path = TEMPORAL / 'audit.json'
    report_path = TEMPORAL / 'REPORT.md'
    records = pd.read_csv(metrics_path).set_index('model', verify_integrity=True)
    expected = ('A_current_hybrid', 'B_temporal')
    assert list(records.index) == list(expected)
    assert records['laqn_station_days'].eq(9619).all()
    assert np.isfinite(records.to_numpy(dtype=float)).all()
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['passed'] and audit['same_rows_for_both_models']
    assert audit['station_rows'] == 9619 and audit['labelled_dates'] == 349
    assert audit['source_inputs_unchanged'] and not audit['selection_used_2024']
    assert {r['variant'] for r in audit['records']} == set(expected)
    for record in audit['records']:
        assert record['metrics_recomputed']
        assert {c['seed'] for c in record['checkpoints']} == {42, 11, 22}
        assert all(re.fullmatch('[0-9a-f]{64}', c['sha256']) for c in record['checkpoints'])
    # Protect against silently rendering scores that no longer match the prose.
    np.testing.assert_allclose(
        records.loc['B_temporal', ['mae_ug_m3', 'rmse_ug_m3', 'r2']].to_numpy(float),
        [2.336123, 3.767902, 0.520059], rtol=0, atol=5e-7,
    )
    report = report_path.read_text(encoding='utf-8')
    assert '[-0.0203, -0.0075]' in report
    names = ['Matched non-temporal hybrid (3 seeds)', 'Temporal hybrid (3 seeds; experimental)']
    table = []
    for key, label in zip(expected, names):
        r = records.loc[key]
        numbers = [f'{r[c]:.4f}' for c in ('mae_ug_m3', 'rmse_ug_m3', 'r2')]
        if key == 'B_temporal':
            numbers = [r'\textbf{' + n + '}' for n in numbers]
        table.append(label + ' & ' + ' & '.join(numbers) + r' \\')
    (ROOT / 'tables/temporal_results.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')
    diagnostics = [
        ('Overall signed bias', 'bias_ug_m3'),
        (r'MAE, observed $<15$', 'mae_below_15'),
        (r'Bias, observed $<15$', 'bias_below_15'),
        (r'MAE, observed $15\leq y<25$', 'mae_15_to_25'),
        (r'MAE, observed $\geq25$', 'mae_at_or_above_25'),
        (r'Bias, observed $\geq25$', 'high_bias_ug_m3'),
    ]
    lines = [label + ' & ' + ' & '.join(f'{records.loc[k, col]:.4f}' for k in expected) + r' \\'
             for label, col in diagnostics]
    (ROOT / 'tables/temporal_diagnostics.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    delta = float(records.loc['B_temporal', 'rmse_ug_m3'] - records.loc['A_current_hybrid', 'rmse_ug_m3'])
    provenance = {
        'source_commit': 'ce40d22',
        'sources': {str(p.relative_to(ROOT.parent)): sha(p) for p in (metrics_path, audit_path, report_path)},
        'station_days': 9619, 'labelled_dates': 349,
        'temporal_minus_control_rmse': delta,
        'relative_rmse_reduction_percent': -100 * delta / float(records.loc['A_current_hybrid', 'rmse_ug_m3']),
        'saved_seven_day_block_rmse_interval': [-0.0203, -0.0075],
        'verification': 'Aggregate transcription checked against the saved CSV/audit/report; prior audit metadata checked. Temporal predictions and checkpoints unavailable locally: no new row-level or checkpoint verification claimed.',
        'temporal_metrics_recomputed_from_predictions_this_revision': False,
        'bootstrap_rerun': False, 'models_retrained': False,
        'model_status': 'Experimental; missed 2023 normal-range bias guard. Original official model unchanged.',
    }
    (ROOT / 'evidence/temporal_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return records, names

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def savefig(fig, name):
    fig.savefig(ROOT / 'figures' / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(ROOT / 'figures' / (name + '.png'), dpi=190, bbox_inches='tight')
    plt.close(fig)

def main():
    global BUNDLE, ART, SOURCE, FINAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE, help='Read-only original London bundle')
    parser.add_argument('--temporal-only', action='store_true', help='Refresh audited temporal aggregate tables only')
    args = parser.parse_args()
    BUNDLE = args.bundle.resolve()
    ART = BUNDLE / 'artifacts'
    SOURCE = ART / 'london_standalone_unet_20260906/test_2024_laqn_all_models.csv.gz'
    FINAL = ART / 'london_3year_final_comparison_20260905_145050/metrics.json'
    for name in ('figures', 'tables', 'evidence'):
        (ROOT / name).mkdir(exist_ok=True)
    temporal, temporal_names = temporal_assets()
    if args.temporal_only:
        print('PASS: temporal aggregates, recorded audit metadata and source hashes checked; no retraining or prediction-level re-audit.')
        return
    d = pd.read_csv(SOURCE)
    assert len(d) == 9619 and d.date.nunique() == 349
    assert not d.duplicated(['date', 'site_code']).any()
    assert d.date.min() == '2024-01-01' and d.date.max() == '2024-12-14'
    y = d.observed_pm25.to_numpy(dtype=float)
    assert np.isfinite(d[['observed_pm25', *MODELS]].to_numpy()).all()
    ref = json.loads((ART / 'london_standalone_unet_20260906/metrics.json').read_text())['models']
    rows = []
    high = y >= 25
    yc = y - d.groupby('date').observed_pm25.transform('mean').to_numpy()
    for key, label in MODELS.items():
        p = d[key].to_numpy(dtype=float)
        e = p-y
        pc = p-d.groupby('date')[key].transform('mean').to_numpy()
        row = dict(model=label, source_column=key, n=len(y), mae=np.abs(e).mean(),
                   rmse=np.sqrt(np.mean(e**2)), r2=1-np.sum(e**2)/np.sum((y-y.mean())**2),
                   bias=e.mean(), spatial_r=np.corrcoef(yc, pc)[0,1],
                   high_n=int(high.sum()), high_mae=np.abs(e[high]).mean(),
                   high_recall=np.mean(p[high]>=25),
                   high_squared_error_percent=100*np.sum(e[high]**2)/np.sum(e**2))
        for metric in ('mae','rmse','r2','bias','high_mae'):
            refkey = 'high_pm25_mae' if metric=='high_mae' else metric
            assert abs(row[metric]-ref[key][refkey]) < 1e-6, (key,metric)
        assert abs(row['spatial_r']-ref[key]['within_day_spatial_r']) < 1e-6
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT/'evidence/recomputed_metrics.csv',index=False)
    lines=[]
    for r in rows:
        nums=[f"{r[k]:.4f}" for k in ('mae','rmse','r2')]
        if r['source_column']=='Residual U-Net (3-seed)':
            nums=[r'\textbf{'+v+'}' for v in nums]
        lines.append(r['model']+' & '+' & '.join(nums)+r' \\')
    (ROOT/'tables/main_results.tex').write_text('\n'.join(lines)+'\n')
    lines=[r['model']+' & '+' & '.join(f"{r[k]:.4f}" for k in ('bias','spatial_r','high_mae'))+r' \\' for r in rows]
    (ROOT/'tables/diagnostics.tex').write_text('\n'.join(lines)+'\n')
    boot=json.loads(FINAL.read_text())['paired_day_bootstrap_vs_winner']
    lines=[]
    for key in ('HGB (unconstrained)','HGB (limited monotonic)','XGBoost','ANN (exact sklearn, 5-seed)'):
        b=boot[key]; lo,hi=b['rmse_difference_95ci']
        lines.append(f"{MODELS[key]} & {b['rmse_difference']:.5f} & [{lo:.5f}, {hi:.5f}]"+r' \\')
    (ROOT/'tables/bootstrap.tex').write_text('\n'.join(lines)+'\n')
    # Complete channel names are copied from the actual model contract.
    names=json.loads((BUNDLE/'data/processed/london_1km_daily/metadata.json').read_text())['feature_names']
    names=[n for n in names if not n.startswith('fire_') and n!='transported_smoke_proxy']
    assert len(names)==66
    feature_lines=[]
    for i,n in enumerate(names):
        escaped=n.replace('_',r'\_')
        feature_lines.append(str(i+1)+r' & \texttt{'+escaped+'}'+r' \\')
    (ROOT/'tables/feature_names.tex').write_text('\n'.join(feature_lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.0,4.4))
    labels=[r['model'].replace('--',' + ') for r in rows] + [n.replace(' (3 seeds', '\n(3 seeds') for n in temporal_names]
    vals=[r['rmse'] for r in rows] + temporal.rmse_ug_m3.tolist()
    positions = list(range(len(rows))) + [len(rows)+.6, len(rows)+1.6]
    ax.barh(positions,vals,color=['#166b74']+['#8ca0af']*6+['#ba966a','#8d5a24'],height=.64)
    ax.axhline(len(rows)-.2, color='#cccccc', lw=.8)
    ax.set_yticks(positions,labels);ax.invert_yaxis()
    ax.set_xlim(0,4.85);ax.set_xlabel('RMSE (µg/m³); lower is better')
    for pos,v in zip(positions,vals):ax.text(v+.04,pos,f'{v:.4f}',va='center')
    ax.set_title('Retrospective 2024: 9,619 LAQN station-days\nBottom pair: separate matched temporal experiment',loc='left',pad=12)
    savefig(fig,'rmse_comparison')
    fig,axs=plt.subplots(1,3,figsize=(9,3.05),sharex=True,sharey=True)
    for ax,key,title in zip(axs,['HGB (unconstrained)','Residual U-Net (3-seed)','Standalone U-Net (3-seed)'],['HGB','Hybrid HGB + U-Net','Standalone U-Net ensemble']):
        ax.hexbin(y,d[key],gridsize=45,extent=(0,65,0,65),mincnt=1,bins='log',cmap='viridis')
        ax.plot([0,65],[0,65],color='#b65339',ls='--',lw=1)
        ax.set(xlim=(0,65),ylim=(0,65),xlabel='Observed PM₂.₅ (µg/m³)',title=title)
        ax.set_aspect('equal')
    axs[0].set_ylabel('Estimated PM₂.₅ (µg/m³)')
    fig.tight_layout();savefig(fig,'parity')
    # A schematic of implemented operations, not a simulated result.
    fig,ax=plt.subplots(figsize=(9,3.6));ax.set_axis_off()
    boxes=[(.02,.58,.25,.31,'Downloaded monitoring\nand environmental\nsource products'),
           (.37,.58,.25,.31,'Quality control, resampling\n66 predictor channels\nObserved cell targets'),
           (.72,.70,.26,.19,'HGB / XGBoost / ANN'),
           (.72,.42,.26,.19,'Standalone U-Net\nDirect estimate; 66 inputs'),
           (.37,.02,.61,.26,'HGB + residual U-Net: original 67 inputs; temporal 72\nTemporal extension adds 5 past-only history summaries\nAverage 3 corrections; scale by 0.425\nInverse target transform → estimated PM₂.₅')]
    for x,y0,w,h,label in boxes:
        ax.add_patch(plt.Rectangle((x,y0),w,h,fc='#eef4f5',ec='#3f6976',lw=1))
        ax.text(x+w/2,y0+h/2,label,ha='center',va='center',fontsize=9)
    for start,end in [((.27,.735),(.37,.735)),((.62,.79),(.72,.79)),((.62,.64),(.72,.515)),((.495,.58),(.495,.28))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#3f6976'})
    ax.set(xlim=(0,1),ylim=(0,1));savefig(fig,'pipeline')
    # Record exactly which source bytes support this paper; no claim of public release.
    manifest={'prediction_file':str(SOURCE.relative_to(BUNDLE)), 'prediction_sha256':sha(SOURCE),
              'bootstrap_file':str(FINAL.relative_to(BUNDLE)),'bootstrap_sha256':sha(FINAL),
              'station_days':len(d),'unique_days':int(d.date.nunique()),'date_min':d.date.min(),'date_max':d.date.max(),
              'high_event_rows':int(high.sum()),'metrics_recomputed':True,'models_retrained':False,
              'checks':'Unique station-day keys, finite observations/predictions, metric agreement < 1e-6; bootstrap intervals transcribed from saved audit, not rerun.'}
    (ROOT/'evidence/provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(metrics[['model','mae','rmse','r2']].to_string(index=False))
    print('PASS: metrics, coverage, feature count and provenance verified; assets generated.')

if __name__=='__main__':
    main()
