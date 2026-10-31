"""Run the planned experiment; --finalize opens the holdout once and exports test predictions."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiment import run_experiment

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',action='store_true',help='Freeze selection, evaluate holdout once, and export test predictions.')
    args=parser.parse_args()
    run_experiment(finalize=args.finalize)
