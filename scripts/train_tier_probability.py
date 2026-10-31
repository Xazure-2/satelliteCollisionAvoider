"""Fit and evaluate P(Safe/Elevated/Critical) for the frozen final-tier model; add them to test event predictions.

--holdout evaluates the probability layer once on holdout forecast events (ledger-guarded).
"""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import pandas as pd
from src.model import FEATURES
from src.final_tier import event_features, predict_events
from src.tier_probability import run, load_probability_model, predict_probabilities

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout',action='store_true')
    args=parser.parse_args()
    result=run(evaluate_holdout=args.holdout)
    print('CV:', {k: round(v['log_loss'],4) for k,v in result['cv'].items()})
    if 'holdout' in result:
        print('Holdout:', {k: round(v['log_loss'],4) for k,v in result['holdout'].items()})
    tier=joblib.load(ROOT/'artifacts/final_tier/final_tier_model.joblib')
    prob=load_probability_model()
    test=pd.read_csv(ROOT/'data/test_data.csv',usecols=['event_id',*FEATURES])
    out=predict_events(test,tier).set_index('event_id')
    p=predict_probabilities(event_features(test),tier,prob)
    out['p_safe'],out['p_elevated'],out['p_critical']=p.Safe,p.Elevated,p.Critical
    out['p_predicted_tier']=[p.loc[e,t] for e,t in zip(out.index,out.predicted_tier)]
    out=out.reset_index()
    out.insert(0,'source','test')
    out.to_csv(ROOT/'outputs/test_event_predictions.csv',index=False)
    print(f'Updated outputs/test_event_predictions.csv ({len(out)} events) with tier probabilities')
