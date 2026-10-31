"""Select the final-tier event model by grouped CV; --finalize evaluates the holdout once. Exports test event predictions."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.final_tier import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',action='store_true',help='Evaluate the frozen selection on the holdout once (ledger-guarded).')
    args=parser.parse_args()
    run(finalize=args.finalize)
