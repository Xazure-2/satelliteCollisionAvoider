"""Fit and evaluate the final-Pc forecast (context for the dashboard; tiers unchanged). --holdout checks it once."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.final_pc import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout',action='store_true')
    result=run(evaluate_holdout=parser.parse_args().holdout)
    for part in ['cv','holdout']:
        if part in result:
            print(part, {k: (round(v['mse_high_risk'],3), v['elevated_or_worse']['recall'] and round(v['elevated_or_worse']['recall'],2)) for k,v in result[part].items()})
