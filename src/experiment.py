"""Small, reproducible experiment with cached validation and a one-time holdout gate."""
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.model import ROOT, FEATURES, PARAMETERS, make_pipeline, predict_log10_pc, prediction_frame
from src.evaluation import evaluate, slice_metrics, bootstrap_events, select_candidate

SELECTION_RULE = ('Among M1/M2/M3 within 1% of the lowest validation raw-log-risk RMSE, maximize '
                  'Pc-critical F2 at -4; break ties by fewer false positives, lower MAE, simpler pipeline, then name.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_partitioned_data(root=ROOT):
    root = Path(root)
    manifest = json.loads((root/'artifacts/eda_manifest.json').read_text())
    settings = manifest['split_settings']
    for source in ['train', 'test']:
        if sha256(root/f'data/{source}_data.csv') != settings[f'{source}_sha256']:
            raise ValueError(f'{source} data changed since the EDA split. Do not silently regenerate splits.')
    if sha256(root/'artifacts/event_splits.csv') != manifest['split_sha256']:
        raise ValueError('Frozen split file was changed.')
    if settings['features'] != FEATURES or settings['seed'] != 42 or settings['fractions'] != [.7,.15,.15]:
        raise ValueError('Unexpected split configuration.')
    splits = pd.read_csv(root/'artifacts/event_splits.csv')
    if not splits.event_id.is_unique or set(splits.partition) != {'train','validation','holdout'}:
        raise ValueError('Split assignments must be unique and cover the three named partitions.')
    data = pd.read_csv(root/'data/train_data.csv', usecols=['event_id', *FEATURES, 'risk'])
    data['row_id'] = np.arange(len(data))
    data = data.loc[np.isfinite(data.time_to_tca) & data.time_to_tca.ge(0) & np.isfinite(data.risk) & data.risk.le(0)].copy()
    if set(splits.event_id) != set(data.event_id):
        raise ValueError('Frozen split coverage differs from eligible training events.')
    data = data.merge(splits, on='event_id', how='left', validate='many_to_one').sort_values('row_id')
    return data, splits, manifest


def compatibility_checks(bundle, frame, artifact_path):
    sample = frame.head(12).copy()
    expected = predict_log10_pc(sample, bundle)
    np.testing.assert_array_equal(expected, predict_log10_pc(sample.drop(columns='risk', errors='ignore'), bundle))
    poisoned = sample.assign(risk=12345., event_id=-999, mission_id=777)
    np.testing.assert_array_equal(expected, predict_log10_pc(poisoned, bundle))
    np.testing.assert_array_equal(expected, predict_log10_pc(sample[sample.columns[::-1]], bundle))
    np.testing.assert_array_equal(expected, predict_log10_pc(sample, joblib.load(artifact_path)))
    changed = sample.copy()
    changed['c_object_type'] = 'NEVER_SEEN_TYPE'
    changed[['c_sigma_r','t_sigma_r','F10','AP']] = np.nan
    assert np.isfinite(predict_log10_pc(changed, bundle)).all()
    assert np.isfinite(predict_log10_pc(sample.drop(columns=['c_sigma_r','t_sigma_r','F10','AP','c_object_type']), bundle)).all()
    invalid = sample.copy(); invalid.loc[invalid.index[0], 'miss_distance'] = -1
    try:
        predict_log10_pc(invalid, bundle)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid required telemetry was accepted.')
    regressor = bundle['pipeline'].named_steps['regressor']
    assert regressor.early_stopping is False
    return {'passed': True, 'checks': ['target and ID exclusion','column-order invariance','save/load equivalence',
        'unknown category','missing optional values/columns','invalid required input rejection','early stopping disabled']}


def run_experiment(root=ROOT, finalize=False, bootstrap_repetitions=200):
    root = Path(root)
    if bootstrap_repetitions < 1:
        raise ValueError('bootstrap_repetitions must be positive')
    data, splits, eda_manifest = load_partitioned_data(root)
    versions = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib','threadpoolctl']}
    source_hashes = {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('evaluation.py')]}
    identity = dict(data=eda_manifest['split_settings'], split_sha256=eda_manifest['split_sha256'],
                    parameters=PARAMETERS, versions=versions, sources=source_hashes,
                    bootstrap_repetitions=bootstrap_repetitions, selection_rule=SELECTION_RULE)
    run_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    ledger_path = root/'artifacts/holdout_evaluation.json'
    if ledger_path.exists():
        prior = json.loads(ledger_path.read_text())
        if prior['run_id'] != run_id:
            raise RuntimeError('The holdout was already opened for a different frozen experiment. '
                               'Do not overwrite its model or use its findings for automatic retuning.')
    run_dir = root/'artifacts/model_runs'/run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/'manifest.json', dict(identity, run_id=run_id, python=platform.python_version(), units=eda_manifest['units']))
    train = data.loc[data.partition.eq('train')].copy()
    validation = data.loc[data.partition.eq('validation')].copy()
    # Holdout labels stay unused until the selected model and its checks are frozen.
    progress_path = root/'artifacts/model_progress.json'
    def progress(stage):
        write_json(progress_path, dict(run_id=run_id, stage=stage, run_dir=str(run_dir)))
        print(stage, flush=True)
    results = {}
    configs = {}
    started = time.time()

    def candidate(name, config):
        progress(f'{name}: fitting or loading cached model')
        model_path = run_dir/f'{name}.joblib'
        result_path = run_dir/f'{name}_metrics.json'
        if model_path.exists() and result_path.exists():
            bundle = joblib.load(model_path)
            result = json.loads(result_path.read_text())
            if result['artifact_sha256'] != sha256(model_path) or result['config'] != config:
                raise ValueError(f'{name} cache integrity mismatch')
        else:
            start = time.time()
            pipeline = make_pipeline(**config)
            with threadpool_limits(limits=4):
                pipeline.fit(train, train.risk)
                prediction = predict_log10_pc(validation, pipeline)
            metadata = dict(model_version=f'{run_id}-{name}', candidate=name, config=config,
                            feature_order=[c for c in FEATURES if config.get('use_max_risk',True) or c!='max_risk_estimate'],
                            seed=42, versions=versions, units=eda_manifest['units'],
                            split_sha256=eda_manifest['split_sha256'], train_event_ids=sorted(train.event_id.unique().tolist()))
            bundle = {'pipeline':pipeline, 'metadata':metadata}
            joblib.dump(bundle, model_path)
            result = dict(candidate=name, config=config, validation=evaluate(validation,prediction),
                          elapsed_seconds=round(time.time()-start,2), artifact_sha256=sha256(model_path))
            write_json(result_path, result)
            pd.DataFrame({'row_id':validation.row_id,'event_id':validation.event_id,
                          'risk':validation.risk,'predicted_log10_pc_raw':prediction}).to_csv(run_dir/f'{name}_validation_predictions.csv',index=False)
        results[name] = result
        configs[name] = config
        m = result['validation']
        progress(f"{name}: validation RMSE={m['rmse']:.4f}, critical recall={m['pc_critical']['recall']}, F2={m['pc_critical']['f2']}")
        return bundle

    try:
        candidate('B0', {'dummy':'median'})
        candidate('B1', {'dummy':'mean'})
        candidate('M1', {'imputation':'median'})
        candidate('M2', {'imputation':'native'})
        parent = select_candidate(results,['M1','M2'])
        candidate('M3', dict(configs[parent], derived=True))
        selected = select_candidate(results,['M1','M2','M3'])
        candidate('A0', dict(configs[selected], use_max_risk=False))
        bundle = joblib.load(run_dir/f'{selected}.joblib')
        checks = compatibility_checks(bundle, train, run_dir/f'{selected}.joblib')
        selected_prediction = pd.read_csv(run_dir/f'{selected}_validation_predictions.csv').predicted_log10_pc_raw.to_numpy()
        detail_path = run_dir/'validation_detail.json'
        if not detail_path.exists():
            progress('Computing validation slices and whole-event bootstrap intervals')
            write_json(detail_path, {'slices':slice_metrics(validation,selected_prediction),
                                    'bootstrap':bootstrap_events(validation,selected_prediction,bootstrap_repetitions)})
        selection = dict(run_id=run_id, selected=selected, derived_parent=parent, selection_rule=SELECTION_RULE,
                         selected_artifact_sha256=sha256(run_dir/f'{selected}.joblib'), checks=checks,
                         baseline_beaten=results[selected]['validation']['rmse'] < results['B1']['validation']['rmse'])
        selection_path = run_dir/'selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError('Frozen selection differs; refusing to overwrite it.')
        write_json(selection_path, selection)
        write_json(run_dir/'comparison.json',results)
        # Copy the selected trained pipeline; no refitting on validation or holdout.
        joblib.dump(bundle, root/'artifacts/selected_model.joblib')
        write_json(root/'artifacts/selected_model_metadata.json',dict(bundle['metadata'], selection=selection))
        fixture = train.head(8)[['event_id', *FEATURES]].copy()
        fixture_dir = root/'tests/fixtures'; fixture_dir.mkdir(parents=True,exist_ok=True)
        fixture.to_csv(fixture_dir/'telemetry.csv',index=False)
        prediction_frame(fixture,bundle,source='training_fixture').to_csv(fixture_dir/'expected_predictions.csv',index=False)
        holdout_result = None
        if finalize:
            progress('Selected model frozen; checking one-time holdout gate')
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                if ledger['run_id'] != run_id or ledger['selected_artifact_sha256'] != selection['selected_artifact_sha256']:
                    raise RuntimeError('Holdout has already been opened for another model/run. No automatic re-evaluation is allowed.')
                if ledger['state'] != 'completed':
                    raise RuntimeError('A prior holdout evaluation did not complete. Inspect the ledger before explicit recovery.')
                holdout_result = json.loads((run_dir/'holdout.json').read_text())
                if sha256(run_dir/'holdout.json') != ledger['result_sha256']:
                    raise ValueError('Saved holdout report integrity mismatch')
                progress('Reusing saved holdout report; no holdout predictions recomputed')
            else:
                ledger = dict(state='started', run_id=run_id,
                              selected_artifact_sha256=selection['selected_artifact_sha256'],
                              split_sha256=eda_manifest['split_sha256'])
                with ledger_path.open('x') as f:
                    json.dump(ledger,f,indent=2)
                holdout = data.loc[data.partition.eq('holdout')].copy()
                with threadpool_limits(limits=4):
                    holdout_prediction = predict_log10_pc(holdout,bundle)
                holdout_result = {'metrics':evaluate(holdout,holdout_prediction),
                    'slices':slice_metrics(holdout,holdout_prediction),
                    'bootstrap':bootstrap_events(holdout,holdout_prediction,bootstrap_repetitions)}
                write_json(run_dir/'holdout.json',holdout_result)
                pd.DataFrame({'row_id':holdout.row_id,'event_id':holdout.event_id,
                    'risk':holdout.risk,'predicted_log10_pc_raw':holdout_prediction}).to_csv(run_dir/'holdout_predictions.csv',index=False)
                write_json(ledger_path,dict(ledger,state='completed',result_sha256=sha256(run_dir/'holdout.json')))
            progress('Exporting test predictions without loading test labels')
            test = pd.read_csv(root/'data/test_data.csv',usecols=['event_id',*FEATURES])
            test['row_id'] = np.arange(len(test))
            with threadpool_limits(limits=4):
                predictions = prediction_frame(test,bundle)
            assert len(predictions)==len(test) and np.array_equal(predictions.event_id,test.event_id)
            outputs = root/'outputs'; outputs.mkdir(exist_ok=True)
            predictions.to_csv(outputs/'test_predictions.csv',index=False)
            predictions[['row_id','event_id','predicted_log10_pc_raw']].rename(
                columns={'predicted_log10_pc_raw':'predicted_risk'}).to_csv(outputs/'test_submission.csv',index=False)
        elif ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if ledger['state'] != 'completed' or ledger['selected_artifact_sha256'] != selection['selected_artifact_sha256']:
                raise RuntimeError('Holdout ledger is incomplete or inconsistent; inspect it before explicit recovery.')
            if sha256(run_dir/'holdout.json') != ledger['result_sha256']:
                raise ValueError('Saved holdout report integrity mismatch')
            holdout_result = json.loads((run_dir/'holdout.json').read_text())
        details = json.loads(detail_path.read_text())
        report = build_report(selection,results,details,holdout_result,splits,data,run_dir,versions)
        (root/'reports').mkdir(exist_ok=True)
        (root/'reports/model_report.md').write_text(report)
        progress(f"Complete: {selected} selected; holdout {'completed/cached' if holdout_result else 'not opened'}; elapsed {time.time()-started:.1f}s")
        return dict(run_dir=str(run_dir), selection=selection, comparison=results,
                    validation_detail=details, holdout=holdout_result)
    except Exception as exc:
        progress(f'FAILED: {type(exc).__name__}: {exc}')
        raise


def fmt(value):
    if value is None: return 'undefined'
    if isinstance(value,float): return f'{value:.4f}'
    return str(value)


def markdown_table(rows):
    if not rows: return '(no rows)'
    keys = list(rows[0])
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+
        ['| '+' | '.join(fmt(row[k]) for k in keys)+' |' for row in rows])


def slice_table(slices):
    return markdown_table([dict(slice=name, rows=m['rows'],events=m['events'],MAE=m['mae'],RMSE=m['rmse'],
        positives=m['pc_critical']['positives'],recall=m['pc_critical']['recall'],F2=m['pc_critical']['f2']) for name,m in slices.items()])


def bootstrap_table(bootstrap):
    return markdown_table([dict(metric=k,**v) for k,v in bootstrap['intervals'].items()])


def metric_section(m):
    return (markdown_table([{'level':'rows', 'rows/events':m['rows'],**m['pc_critical']},
        {'level':'latest event','rows/events':m['latest_pc']['events'],**m['latest_pc']['pc_critical']}])+
        '\n\nLatest-message Pc false alerts per 1,000 events: '+fmt(m['latest_pc']['false_alerts_per_1000_events'])+
        '\n\nLatest-message combined-policy results:\n\n'+markdown_table([m['latest_combined']])+
        '\n\nConfusion matrices: rows = truth, columns = prediction; order '+str(m['tier_labels'])+
        '. The fourth category is reserved for invalid inputs.\n\nPc only:\n\n```text\n'+
        np.array2string(np.asarray(m['pc_confusion_matrix']))+'\n```\n\nCombined policy:\n\n```text\n'+
        np.array2string(np.asarray(m['combined_confusion_matrix']))+'\n```')


def build_report(selection, results, details, holdout, splits, data, run_dir, versions):
    selected = selection['selected']; selected_m = results[selected]['validation']
    rows = [dict(model=name,imputation=r['config'].get('imputation','median'),derived=r['config'].get('derived',False),
        RMSE=r['validation']['rmse'],MAE=r['validation']['mae'],recall=r['validation']['pc_critical']['recall'],
        F2=r['validation']['pc_critical']['f2'],AP=r['validation']['pc_critical']['average_precision'],
        FP=r['validation']['pc_critical']['fp'],FN=r['validation']['pc_critical']['fn']) for name,r in results.items()]
    counts=[]
    for name in ['train','validation','holdout'] if holdout else ['train','validation']:
        f=data.loc[data.partition.eq(name)]
        counts.append(dict(partition=name,rows=len(f),events=f.event_id.nunique(),Pc_critical_rows=int(f.risk.ge(-4).sum()),
                           Pc_critical_events=f.loc[f.risk.ge(-4),'event_id'].nunique()))
    report = f'''# Row-level collision-risk model report

## Result and decision

Selected **{selected}**, model version `{selection['run_id']}-{selected}`, using validation only. M3 inherited preprocessing from **{selection['derived_parent']}**. The selected model {'beats' if selection['baseline_beaten'] else 'does not beat'} the mean-target dummy's validation RMSE. No held-out data were used to fit preprocessing or the estimator.

{SELECTION_RULE}

The 1% tolerance is our practical balanced-selection default, not an organizer-supplied scoring formula. A0 is diagnostic only and never participates in selection. Scores use **raw** predicted log risk; displayed Pc uses `10 ** clip(raw, -30, 0)`. No oversampling, class weighting, target-floor removal, log transforms, PCA, or automatic row-based early stopping was used.

## Frozen data split and validation experiment

{markdown_table(counts)}

{markdown_table(rows)}

B0 = median dummy; B1 = mean dummy; M1 = median-imputed HGB; M2 = native-NaN HGB; M3 = selected preprocessing plus geometry features; A0 = selected architecture without `max_risk_estimate`.

All learned models use the same two extreme-sigma flags and numeric missingness indicators. Categories are one-hot encoded from training only; unknown categories are accepted. Derived features are absolute radial position, independent-error combined radial sigma, and their ratio (missing for invalid/zero denominator), calculated before imputation. Extreme sigmas remain measurements, not automatically discarded sentinels.

Parameters: `{json.dumps(PARAMETERS,sort_keys=True)}`.

## Selected-model validation diagnostics

{metric_section(selected_m)}

### Slices (overlap; do not sum their row counts)

{slice_table(details['slices'])}

### Event-cluster bootstrap

{bootstrap_table(details['bootstrap'])}

{details['bootstrap']['repetitions']} resamples, seed 42, whole events sampled with replacement and all their rows retained. Percentile 95% intervals describe sampling variation conditional on this trained model. They do not cover model-training uncertainty, physical Pc uncertainty, or distribution shift. Undefined replicate counts are explicit. Average precision is computed from continuous raw log-risk scores; it is not trapezoidal PR area. Precision is undefined with no positive predictions; recall/F2/AP are undefined with no actual positives.

'''
    if holdout:
        report += f'''## Single holdout evaluation

The selection and model hash were frozen and compatibility checks passed before this evaluation. Reruns reuse the saved result; the holdout ledger blocks automatic evaluation of another model. These findings must not initiate a tuning cycle.

MAE: **{holdout['metrics']['mae']:.4f}**; RMSE: **{holdout['metrics']['rmse']:.4f}**.

{metric_section(holdout['metrics'])}

### Holdout slices

{slice_table(holdout['slices'])}

### Holdout event-cluster bootstrap

{bootstrap_table(holdout['bootstrap'])}

'''
    else:
        report += '## Holdout\n\nNot opened. Run the explicit finalization command only after reviewing validation and checks.\n\n'
    report += f'''## Interpretation and limitations

- The target is each message's calculated log10(Pc), not an observed collision outcome or final-event forecast. Exponentiating a mean log-risk estimate does not generally yield mean physical Pc.
- The −30 floor remains in training. Consult non-floor and high-risk slices before drawing conclusions from aggregate errors.
- The latest-event positive sample is sparse; recall must be read with its actual denominator and bootstrap undefined counts. Combined-rule performance includes direct distance triggers and is not a measure of Pc estimation alone.
- `max_risk_estimate` is an explicitly permitted upstream risk-derived feature. Compare A0 with the selected model to see dependence on it; this diagnostic alone does not establish leakage.
- The EDA found no permitted-telemetry overlap candidates, but its rounded screen cannot prove independent physical encounters. IDs are source-local and never predictors.
- Test telemetry begins more than two days before TCA. Validation's `test_horizon_ge_2_days` slice is particularly relevant; no test labels were loaded or evaluated.
- Low critical recall would motivate a separately validated alert classifier in a later experiment, not silent threshold changes, inflated Pc values, or a hard low-risk gate. No auxiliary classifier was trained here.
- This model does not reconstruct complete covariance, physical collision probability, or maneuver commands; no calibration or operational-certification claim is made.

## Reproduction and artifacts

Run `python scripts/train.py` for cached validation experiments; add `--finalize` for the one-time holdout evaluation and test export. `python scripts/predict.py` regenerates exports with the frozen selected model. The train command reuses the EDA split and validates raw-file fingerprints. It never refits the selected model on validation/holdout.

Run directory: `{run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}`. It contains candidate pipelines, validation predictions, detailed metrics, the frozen selection, and the manifest. Stable model: `artifacts/selected_model.joblib`; metadata: `artifacts/selected_model_metadata.json`. Handoff fixture: `tests/fixtures/telemetry.csv` and its expected predictions. Test exports preserve original row order and IDs; `test_submission.csv` contains `row_id,event_id,predicted_risk` where predicted risk is raw log10(Pc). Confirm the organizer's exact header before submission.

Checks: {', '.join(selection['checks']['checks'])}. Dependency versions: `{json.dumps(versions,sort_keys=True)}`. The manifest records data/source/split hashes and the model metadata includes feature order, units, seed, and training event IDs. Joblib files must come from a trusted source.

## Research informing this experiment

- [ESA challenge analysis](https://www.honda-ri.de/pubs/pdf/4688.pdf): strong baselines, tree models, and evaluation pitfalls; its final-event task differs from ours.
- [Supervised learning with missing values](https://arxiv.org/abs/1902.06931): motivates comparing simple imputation with direct missing-value handling.
- [Tabular-model benchmark](https://arxiv.org/abs/2207.08815): supports a boosted-tree starting point, without claiming universal superiority.
- [Satellite two-stage models](https://www.sciencedirect.com/science/article/pii/S246889672100094X): informs possible future alert classification, not a result claimed for our task.
- [NASA risk assessment](https://www.nasa.gov/cara/step-2-close-approach-risk-assessment/): physical Pc requires more than our permitted telemetry subset.
'''
    return report
