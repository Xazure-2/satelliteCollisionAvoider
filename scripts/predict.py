"""Per-message estimated Pc and tiers for the dashboard; never loads risk or other excluded columns."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model import ROOT, FEATURES, OPTIONAL, load_bundle, prediction_frame
from src.row_model import ARTIFACT

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/test_data.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/test_predictions.csv')
    parser.add_argument('--model',type=Path,default=ARTIFACT,help='Row-level Pc model (default: experiment v2 winner).')
    args=parser.parse_args()
    frame=pd.read_csv(args.input,usecols=lambda c:c in ['event_id',*FEATURES])
    frame['row_id']=range(len(frame))
    for col in OPTIONAL:
        if col not in frame: frame[col]=None
    source='test' if args.input.resolve() == (ROOT/'data/test_data.csv').resolve() else args.input.stem
    with threadpool_limits(limits=4):
        bundle=load_bundle(args.model)
        predictions=prediction_frame(frame,bundle,source=source)
    floor=bundle['metadata'].get('floor')
    if floor is not None:
        # Floored estimates mean "below 1e-6" (Safe by Pc), not an exact probability.
        at_floor=predictions.predicted_log10_pc_raw.le(floor+1e-9)
        predictions.insert(predictions.columns.get_loc('predicted_pc')+1,'pc_display',
            predictions.predicted_pc.map(lambda p:f'{p:.1e}').where(~at_floor,'< 1e-6').where(predictions.predicted_pc.notna(),'needs review'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False)
    print(f'Exported {len(predictions):,} predictions to {args.output}')
