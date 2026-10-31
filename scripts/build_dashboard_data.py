"""Write dashboard-ready tables: test events (demo) and a held-out training replay with observed outcomes.

The replay uses holdout events only, so both models' estimates are out of sample. Observed final
tiers come from train_data labels and are kept in separate `observed_*` columns, never used as input.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model import ROOT, FEATURES
from src.risk import to_csv_ready
from src.row_model import load_row_model
from src.events import build_tables
from src.final_tier import final_labels


def write(tables, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    to_csv_ready(tables['rows']).to_csv(out_dir/f'{name}_rows.csv', index=False)
    queue = tables['queue'].copy()
    queue['reasons'] = [json.dumps(r) for r in queue.reasons]
    to_csv_ready(queue).to_csv(out_dir/f'{name}_events.csv', index=False)
    (out_dir/f'{name}_summary.json').write_text(json.dumps(dict(counts=tables['counts'], insights=tables['insights']), indent=2)+'\n')
    print(f"{name}: {len(tables['rows']):,} messages, {len(tables['events']):,} events; window counts {tables['counts']}")


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff-days',type=float,default=0.0,help='Replay what was known this many days before each TCA (training replay only).')
    parser.add_argument('--window-days',type=float,default=1.0)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/dashboard')
    args=parser.parse_args()
    row_bundle=load_row_model()
    tier_bundle=joblib.load(ROOT/'artifacts/final_tier/final_tier_model.joblib')  # trusted local artifact
    with threadpool_limits(limits=4):
        test=pd.read_csv(ROOT/'data/test_data.csv',usecols=['event_id',*FEATURES])
        write(build_tables(test,row_bundle,tier_bundle,source='test',window_days=args.window_days),args.out,'test')
        train=pd.read_csv(ROOT/'data/train_data.csv',usecols=['event_id',*FEATURES,'risk'])
        train['row_id']=range(len(train))
        splits=pd.read_csv(ROOT/'artifacts/event_splits.csv')
        holdout=train.loc[train.event_id.isin(splits.loc[splits.partition.eq('holdout'),'event_id'])]
        labels=final_labels(holdout).reset_index()
        outcomes=pd.DataFrame({'source':'train_replay','event_id':labels.event_id,
            'observed_final_tier':labels.final_tier.str.lower(),'observed_final_miss_distance_m':labels.final_miss_distance,
            'observed_final_time_to_tca_days':labels.final_time_to_tca})
        replay=build_tables(holdout.drop(columns='risk'),row_bundle,tier_bundle,source='train_replay',
                            cutoff_days=args.cutoff_days,outcomes=outcomes,window_days=args.window_days)
        write(replay,args.out,'train_replay')
