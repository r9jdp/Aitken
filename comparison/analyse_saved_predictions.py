"""Read existing predictions without retraining; write aggregate evidence in comparison only."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def analyse(source):
    before = sha(source)
    with gzip.open(source, 'rt', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        r['y'] = float(r['observed_pm25'])
        r['p'] = float(r['Residual U-Net (3-seed)'])
        assert math.isfinite(r['y']) and math.isfinite(r['p'])
        r['e'] = r['p'] - r['y']
        r['se'] = r['e'] ** 2
    assert len(rows) == 9619
    assert len({(r['date'], r['site_code']) for r in rows}) == len(rows)
    total = sum(r['se'] for r in rows)

    def stats(group):
        n = len(group)
        yy = sum(r['y'] for r in group) / n
        pp = sum(r['p'] for r in group) / n
        sse = sum(r['se'] for r in group)
        sst = sum((r['y'] - yy) ** 2 for r in group)
        return dict(n=n, observed_mean=yy, predicted_mean=pp,
                    mae=sum(abs(r['e']) for r in group)/n,
                    rmse=math.sqrt(sse/n), r2=1-sse/sst if sst else None,
                    bias=pp-yy, row_percent=100*n/len(rows),
                    squared_error_percent=100*sse/total)

    high = [r for r in rows if r['y'] >= 25]
    days, months, changes = defaultdict(list), defaultdict(list), defaultdict(list)
    lookup = {(r['date'], r['site_code']): r for r in rows}
    for r in rows:
        days[r['date']].append(r)
        months[r['date'][:7]].append(r)
        previous = (date.fromisoformat(r['date'])-timedelta(days=1)).isoformat()
        old = lookup.get((previous, r['site_code']))
        if old:
            delta = r['y']-old['y']
            label = 'rise_at_least_10' if delta >= 10 else 'fall_at_least_10' if delta <= -10 else 'change_less_than_10'
            changes[label].append(r)
    result = dict(model='Original hybrid HGB plus residual U-Net three-seed ensemble',
                  scope='Descriptive recomputation of saved 2024 predictions; no model training or selection',
                  input_basename=source.name, input_sha256=before,
                  n_dates=len(days), first_date=min(days), last_date=max(days),
                  overall=stats(rows), below_25=stats([r for r in rows if r['y']<25]),
                  at_least_25=stats(high),
                  months={k:stats(v) for k,v in sorted(months.items())},
                  highest_error_days=sorted([dict(date=k,**stats(v)) for k,v in days.items()],key=lambda x:x['squared_error_percent'],reverse=True)[:10],
                  ordinary_example=dict(date='2024-07-15',**stats(days['2024-07-15'])),
                  observed_day_to_day_change_strata={k:stats(v) for k,v in changes.items()})
    result['at_least_25']['underprediction_count'] = sum(r['e']<0 for r in high)
    result['at_least_25']['predicted_at_least_25_count'] = sum(r['p']>=25 for r in high)
    with (ROOT/'results/metrics.csv').open(newline='') as f:
        published = next(csv.DictReader(f))
    for key, field in [('mae','mae_ug_m3'),('rmse','rmse_ug_m3'),('r2','r2')]:
        assert abs(result['overall'][key]-float(published[field])) < 1e-10
    assert before == sha(source)
    result['verification'] = {'source_unchanged':True,'published_hybrid_metrics_match':True,
                              'all_evaluation_rows_retained':True,'no_station_rows_redistributed':True}
    output = HERE/'evidence'
    output.mkdir(exist_ok=True)
    (output/'high_pollution_diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ['overall','at_least_25','verification']},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True,type=Path)
    analyse(p.parse_args().predictions.resolve())
