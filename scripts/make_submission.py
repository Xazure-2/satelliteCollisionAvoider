"""Write the Challenge #1 submission: one row per test event with its predicted final tier and probability.

Computed directly from the saved models (final-tier model + probability calibrator) and test telemetry;
test `risk` is never read. Output: submission/submission.csv (tracked in git). Columns are described in
submission/README.md.
"""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from src.model import FEATURES
from src.final_tier import event_features, predict_events, TIER_NAMES
from src.tier_probability import load_probability_model, predict_probabilities

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'submission/submission.csv')
    args=parser.parse_args()
    tier=joblib.load(ROOT/'artifacts/final_tier/final_tier_model.joblib')  # trusted local artifacts
    prob=load_probability_model()
    test=pd.read_csv(ROOT/'data/test_data.csv',usecols=['event_id',*FEATURES])
    events=predict_events(test,tier).set_index('event_id')
    p=predict_probabilities(event_features(test),tier,prob).reindex(events.index)
    sub=pd.DataFrame({
        'event_id':events.index,
        'predicted_final_tier':events.predicted_tier.to_numpy(),
        'probability':[p.loc[e,t] for e,t in zip(events.index,events.predicted_tier)],
        'p_safe':p.Safe.to_numpy(),'p_elevated':p.Elevated.to_numpy(),'p_critical':p.Critical.to_numpy(),
        'requires_manoeuvre_planning':events.predicted_tier.eq('Critical').to_numpy(),
        'predicted_final_miss_distance_m':events.predicted_final_miss_distance_m.round(1).to_numpy(),
        'latest_time_to_tca_days':events.latest_time_to_tca_days.round(4).to_numpy(),
        'model_version':events.model_version.to_numpy(),
    }).sort_values('event_id')
    for col in ['probability','p_safe','p_elevated','p_critical']:
        sub[col]=sub[col].round(4)
    # Integrity checks before writing.
    assert sub.event_id.is_unique and set(sub.event_id)==set(test.event_id), 'one row per test event'
    assert sub.predicted_final_tier.isin(TIER_NAMES).all()
    assert np.allclose(sub[['p_safe','p_elevated','p_critical']].sum(axis=1),1,atol=1e-3)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sub.to_csv(args.out,index=False)
    counts=sub.predicted_final_tier.value_counts().reindex(TIER_NAMES[::-1],fill_value=0)
    print(f'Wrote {len(sub):,} events to {args.out}: '+', '.join(f'{k} {v}' for k,v in counts.items()))
