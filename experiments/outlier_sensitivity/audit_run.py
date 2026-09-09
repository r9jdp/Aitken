"""Independent saved-prediction/integrity audit; no training or model selection."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_experiment import inside_repo, json_save, sha256, array_hash
from frozen_helpers import station_rows


def audit(run, reference, output):
    run, output = inside_repo(run), inside_repo(output)
    summary = json.loads((run/'summary.json').read_text())
    integrity = json.loads((run/'input_integrity.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Only a completed run can pass this final audit')
    assert integrity['before'] == integrity['after']
    assert all(sha256(p)==value for p,value in integrity['before'].items())
    config = json.loads((Path(__file__).parent/'frozen_protocol.json').read_text())
    manifest = json.loads((run/'input_manifest.json').read_text())
    data = next(Path(p).parent for p in manifest['sha256'] if Path(p).name == 'station_daily.npz')
    observed_rows = {730:station_rows(data,730,1095), 1095:station_rows(data,1095,1461)}
    # Verify the unchanged backbone and extracted helper bodies against source AST.
    here = Path(__file__).parent
    def parsed(path): return ast.parse(path.read_text(encoding='utf-8-sig'))
    assert ast.dump(parsed(here/'london_unet_model.py')) == ast.dump(parsed(reference/'scripts/london_unet_model.py'))
    copied = {n.name:ast.dump(n) for n in parsed(here/'frozen_helpers.py').body if isinstance(n,ast.FunctionDef)}
    sources = {'train_london_unet.py':['inverse_target'],
               'tune_london_unet_2023.py':['gaussian_kernel','derive_lds_lookup'],
               'train_london_standalone_unet.py':['fit_preprocessing','station_rows','gather']}
    for name, functions in sources.items():
        original={n.name:ast.dump(n) for n in parsed(reference/'scripts'/name).body if isinstance(n,ast.FunctionDef)}
        for f in functions: assert copied[f] == original[f], f
    records=[]
    for family, treatments in summary['screening'].items():
        d=run/'screening'/family
        reference_rows=None
        init=[]
        for name,record in treatments.items():
            completed=json.loads((d/name/'completed.json').read_text())
            assert sha256(d/name/'final_ema_checkpoint.pt')==completed['checkpoint_sha256']
            assert completed['epochs']==len(config[family+'_schedules']['42'])
            init.append(completed['initial_parameter_hash'])
            frame=pd.read_csv(d/f'{name}_predictions.csv.gz')
            identity=frame.drop(columns='predicted_pm25')
            pd.testing.assert_frame_equal(identity,observed_rows[730],check_dtype=False,rtol=1e-6,atol=2e-6)
            if reference_rows is None: reference_rows=identity
            else: pd.testing.assert_frame_equal(reference_rows,identity)
            y=frame.observed_pm25.to_numpy(float)
            p=frame.predicted_pm25.to_numpy(float)
            errors=p-y
            measured={'rmse':np.sqrt(np.mean(errors**2)), 'mae':np.mean(abs(errors)),
                      'r2':1-np.sum(errors**2)/np.sum((y-y.mean())**2),'bias':errors.mean()}
            for key,value in measured.items(): assert abs(value-record['metrics'][key])<1e-7
            high_mae=float(np.mean(abs(errors[y>=25])))
            assert abs(high_mae-record['metrics']['high']['mae'])<1e-7
            maps=np.load(d/f'{name}_maps.npy',mmap_mode='r')
            np.testing.assert_allclose(maps[frame.date_idx.to_numpy(int)-730,frame.row.to_numpy(int),frame.col.to_numpy(int)],p,rtol=1e-6,atol=2e-6)
            records.append({'family':family,'treatment':name,'rows':len(y),'high_rows':int((y>=25).sum()),'checkpoint_verified':True,'metrics_recomputed':True,'map_station_alignment':True})
        assert len(set(init))==1
        control=treatments['control']['metrics']
        eligible=[]
        for name,r in treatments.items():
            m=r['metrics']
            if name!='control' and m['rmse']<control['rmse'] and m['mae']<=control['mae'] and m['high']['mae']<=control['high']['mae']:
                eligible.append(name)
        winner=min(eligible,key=lambda n:treatments[n]['metrics']['rmse']) if eligible else None
        assert summary['selection'][family]['selected']==winner
        if winner is None: assert family not in summary['final']
    for family,results in summary['final'].items():
        d=run/'final'/family
        original=None
        paired_initial={seed:[] for seed in [42,11,22]}
        for treatment,m in results.items():
            if treatment=='paired_block_interval': continue
            f=pd.read_csv(d/f'{treatment}_predictions.csv.gz')
            pd.testing.assert_frame_equal(f.drop(columns='predicted_pm25'),observed_rows[1095],check_dtype=False,rtol=1e-6,atol=2e-6)
            if original is None: original=f.drop(columns='predicted_pm25')
            else: pd.testing.assert_frame_equal(original,f.drop(columns='predicted_pm25'))
            e=f.predicted_pm25.to_numpy(float)-f.observed_pm25.to_numpy(float)
            assert abs(np.sqrt(np.mean(e**2))-m['rmse'])<1e-7
            y=f.observed_pm25.to_numpy(float)
            measured={'mae':np.mean(abs(e)), 'bias':e.mean(),
                      'r2':1-np.sum(e**2)/np.sum((y-y.mean())**2)}
            for key,value in measured.items(): assert abs(value-m[key])<1e-7
            for key, value in {'mae':np.mean(abs(e[y>=25])), 'rmse':np.sqrt(np.mean(e[y>=25]**2)), 'bias':e[y>=25].mean()}.items():
                assert abs(value-m['high'][key])<1e-6
            maps=np.load(d/f'{treatment}_maps.npy',mmap_mode='r')
            frozen=json.loads((d/'prediction_freeze.json').read_text())
            assert array_hash(maps)==frozen[treatment]
            np.testing.assert_allclose(maps[f.date_idx.to_numpy(int)-1095,f.row.to_numpy(int),f.col.to_numpy(int)],f.predicted_pm25,rtol=1e-6,atol=2e-6)
            for seed in [42,11,22]:
                completed=json.loads((d/treatment/f'seed_{seed}'/'completed.json').read_text())
                assert sha256(d/treatment/f'seed_{seed}'/'final_ema_checkpoint.pt')==completed['checkpoint_sha256']
                assert completed['epochs']==len(config[family+'_schedules'][str(seed)])
                paired_initial[seed].append(completed['initial_parameter_hash'])
        assert all(len(set(values))==1 for values in paired_initial.values())
        records.append({'family':family,'phase':'2024_confirmation','identical_observations':True,'checkpoint_verified':True,'metrics_recomputed':True,'prediction_freeze_verified':True,'map_station_alignment':True})
    result={'status':'PASS','all_input_hashes_still_unchanged':True,'backbone_and_helpers_match_original_source':True,
            'paired_initialisation_verified':True,'evaluation_observations_match_original_station_archive':True,'selection_gate_independently_recomputed':True,'trials':records}
    json_save(output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--reference-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit(args.run_dir,args.reference_bundle,args.output)
