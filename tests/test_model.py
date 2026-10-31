import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model import FEATURES, PARAMETERS, TelemetryFeatures, make_pipeline, predict_log10_pc, risk_tier, latest_indices
from src.evaluation import detection, bootstrap_events, select_candidate


def fixture():
    n=80
    rng=np.random.default_rng(42)
    frame=pd.DataFrame({c:rng.uniform(1,10,n) for c in FEATURES if c!='c_object_type'})
    frame['c_object_type']=np.tile(['DEBRIS','PAYLOAD'],n//2)
    frame['event_id']=np.repeat(np.arange(n//4),4)
    frame['row_id']=np.arange(n)
    frame['risk']=-10 + frame.miss_distance/2
    frame.loc[[1,3],'c_sigma_r']=np.nan
    frame.loc[[5,9],['F10','AP']]=np.nan
    return frame


class ModelTests(unittest.TestCase):
    def test_feature_derivation_before_imputation_and_no_label(self):
        x=fixture(); x.loc[0,['c_sigma_r','t_sigma_r']]=[3.,4.]
        x.loc[2,['c_sigma_r','t_sigma_r']]=[0.,0.]
        x.loc[4,'c_sigma_r']=63781363.
        t=TelemetryFeatures(derived=True).fit_transform(x)
        self.assertEqual(t.loc[0,'combined_radial_sigma'],5)
        self.assertTrue(np.isnan(t.loc[1,'combined_radial_sigma']))
        self.assertTrue(np.isnan(t.loc[2,'radial_sigma_ratio']))
        self.assertEqual(t.loc[4,'c_sigma_extreme'],1)
        self.assertEqual(t.loc[4,'c_sigma_r'],63781363.)
        self.assertNotIn('risk',t); self.assertNotIn('event_id',t)

    def test_pipeline_imputation_inference_and_serialization(self):
        x=fixture()
        for mode in ['median','native']:
            with self.subTest(mode=mode), threadpool_limits(limits=1):
                model=make_pipeline(imputation=mode,derived=True,parameters=dict(PARAMETERS,max_iter=3,min_samples_leaf=2))
                model.fit(x,x.risk)
                p=predict_log10_pc(x,model)
                np.testing.assert_array_equal(p,predict_log10_pc(x.assign(risk=999,event_id=-1),model))
                np.testing.assert_array_equal(p,predict_log10_pc(x.drop(columns='risk')[x.drop(columns='risk').columns[::-1]],model))
                altered=x.copy(); altered['c_object_type']='new'; altered[['c_sigma_r','t_sigma_r','F10','AP']]=np.nan
                self.assertTrue(np.isfinite(predict_log10_pc(altered,model)).all())
                self.assertTrue(np.isfinite(predict_log10_pc(x.drop(columns=['F10','AP','c_sigma_r','t_sigma_r','c_object_type']),model)).all())
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'model.joblib'; joblib.dump(model,path)
                    np.testing.assert_array_equal(p,predict_log10_pc(x,joblib.load(path)))
                self.assertFalse(model.named_steps['regressor'].early_stopping)
                if mode=='median':
                    imp=model.named_steps['preprocess'].named_transformers_['numeric']
                    names=model.named_steps['preprocess'].transformers_[0][2]
                    self.assertEqual(imp.statistics_[names.index('c_sigma_r')],x.c_sigma_r.median())
                invalid=x.copy(); invalid.loc[0,'miss_distance']=-1
                with self.assertRaises(ValueError): predict_log10_pc(invalid,model)

    def test_ablation_drops_max_risk(self):
        x=fixture()
        with threadpool_limits(limits=1):
            model=make_pipeline(use_max_risk=False,parameters=dict(PARAMETERS,max_iter=2))
            model.fit(x,x.risk)
            np.testing.assert_array_equal(model.predict(x),model.predict(x.assign(max_risk_estimate=1e10)))

    def test_thresholds_and_event_ties(self):
        self.assertEqual(risk_tier([-7,-7,-7,-6,-4,np.nan,np.nan],[5001,5000,1000,6000,6000,6000,999]).tolist(),
                         ['safe','elevated','elevated','elevated','critical','needs_review','critical'])
        x=pd.DataFrame({'source':['train','train','train','test'], 'event_id':[1]*4,'time_to_tca':[2,1,1,.5],'row_id':[0,1,2,0]})
        self.assertEqual(set(latest_indices(x)),{2,3})

    def test_undefined_metrics_and_cluster_bootstrap(self):
        self.assertIsNone(detection([False,False],[-9,-8])['recall'])
        self.assertIsNone(detection([True,False],[-9,-8])['precision'])
        x=fixture(); x['risk']=-8
        boot=bootstrap_events(x,np.full(len(x),-8.),repetitions=12)
        self.assertEqual(boot['intervals']['pc_recall']['undefined_replicates'],12)
        self.assertEqual(boot['intervals']['rmse']['high_95'],0)
        # Identical rows within a group are retained together, not sampled independently.
        x=pd.DataFrame({'event_id':[0,0,1], 'time_to_tca':[2,1,1], 'row_id':[0,1,2], 'risk':[-8.,-8.,-8.]})
        boot=bootstrap_events(x,np.array([-7.,-7.,-5.]),repetitions=100)
        self.assertGreaterEqual(boot['intervals']['mae']['low_95'],1)
        self.assertLessEqual(boot['intervals']['mae']['high_95'],3)
        # Each sampled event brings all its rows; unequal group sizes change the denominator.
        draw=np.random.default_rng(42).integers(0,2,2)
        absolute=np.array([2.,3.])[draw].sum()
        count=np.array([2.,1.])[draw].sum()
        single=bootstrap_events(x,np.array([-7.,-7.,-5.]),repetitions=1,seed=42)
        self.assertEqual(single['intervals']['mae']['low_95'],absolute/count)

    def test_selection_gate(self):
        def item(rmse,f2,fp,mae): return {'validation':{'rmse':rmse,'mae':mae,'pc_critical':{'f2':f2,'fp':fp}}}
        results={'M1':item(1.,.2,10,.6),'M2':item(1.005,.3,9,.7),'M3':item(1.02,.9,1,.5)}
        self.assertEqual(select_candidate(results,list(results)),'M2')

    def test_holdout_gate_blocks_changed_run_before_training(self):
        from src.experiment import run_experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'artifacts').mkdir()
            (root/'artifacts/holdout_evaluation.json').write_text(json.dumps({'run_id':'another-frozen-run'}))
            manifest={'split_settings':{},'split_sha256':'frozen'}
            with patch('src.experiment.load_partitioned_data',return_value=(None,None,manifest)), \
                 patch('src.experiment.make_pipeline') as fit_factory:
                with self.assertRaisesRegex(RuntimeError,'holdout was already opened'):
                    run_experiment(root,finalize=True)
                fit_factory.assert_not_called()

if __name__=='__main__': unittest.main()
