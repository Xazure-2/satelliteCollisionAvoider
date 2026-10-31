"""Write outputs/event_predictions.csv and outputs/cdm_history.csv for the Reflex dashboard (v2 contract).

Test events: predictions only (test `risk` is never read). Held-out training events: out-of-sample
predictions plus observed risk history and actual final outcomes for the Model Validation views.
The files are checked with the dashboard's own loader before the script reports success.
"""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model import FEATURES
from src.row_model import load_row_model
from src.final_tier import final_labels
from src.dashboard_export import export_tables
from src.tier_probability import load_probability_model
from src.final_pc import load_final_pc_model

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs')
    parser.add_argument('--no-holdout',action='store_true',help='Export test events only.')
    args=parser.parse_args()
    row_bundle=load_row_model()
    tier_bundle=joblib.load(ROOT/'artifacts/final_tier/final_tier_model.joblib')  # trusted local artifact
    prob_bundle=load_probability_model()
    final_pc_bundle=load_final_pc_model()
    parts=[]
    with threadpool_limits(limits=4):
        test=pd.read_csv(ROOT/'data/test_data.csv',usecols=['event_id',*FEATURES])
        test['row_id']=range(len(test))
        parts.append(export_tables(test,row_bundle,tier_bundle,source='test',split='test',prob_bundle=prob_bundle,final_pc_bundle=final_pc_bundle))
        if not args.no_holdout:
            train=pd.read_csv(ROOT/'data/train_data.csv',usecols=['event_id',*FEATURES,'risk'])
            train['row_id']=range(len(train))
            splits=pd.read_csv(ROOT/'artifacts/event_splits.csv')
            holdout=train.loc[train.event_id.isin(splits.loc[splits.partition.eq('holdout'),'event_id'])]
            parts.append(export_tables(holdout.drop(columns='risk'),row_bundle,tier_bundle,source='train',split='holdout',
                                       observed_risk=holdout.risk,labels=final_labels(holdout),prob_bundle=prob_bundle,final_pc_bundle=final_pc_bundle))
    events=pd.concat([e for e,_ in parts],ignore_index=True)
    history=pd.concat([h for _,h in parts],ignore_index=True)
    args.out.mkdir(parents=True,exist_ok=True)
    events.to_csv(args.out/'event_predictions.csv',index=False)
    history.to_csv(args.out/'cdm_history.csv',index=False)
    sys.path.insert(0,str(ROOT/'dashboard'))
    from orbital.data import adapters
    dataset=adapters.load_real(args.out)
    print(f'Wrote {len(events):,} events and {len(history):,} CDMs to {args.out}')
    print('Dashboard loader: OK;', dataset.model_version, '| warnings:', dataset.warnings or 'none')
    print(events.groupby(['source','predicted_final_tier']).size().unstack(fill_value=0).to_string())
