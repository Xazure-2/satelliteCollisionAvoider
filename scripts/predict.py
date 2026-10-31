"""Predict permitted row telemetry without loading risk or other excluded columns."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model import ROOT, FEATURES, OPTIONAL, load_bundle, prediction_frame

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/test_data.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/test_predictions.csv')
    parser.add_argument('--model',type=Path,default=ROOT/'artifacts/selected_model.joblib')
    args=parser.parse_args()
    frame=pd.read_csv(args.input,usecols=lambda c:c in ['event_id',*FEATURES])
    for col in OPTIONAL:
        if col not in frame: frame[col]=None
    source='test' if args.input.resolve() == (ROOT/'data/test_data.csv').resolve() else args.input.stem
    with threadpool_limits(limits=4):
        predictions=prediction_frame(frame,load_bundle(args.model),source=source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False)
    print(f'Exported {len(predictions):,} predictions to {args.output}')
