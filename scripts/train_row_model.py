"""Fit the production per-message Pc model (experiment v2 winner) on train+validation events."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.row_model import train, ARTIFACT

if __name__ == '__main__':
    bundle=train()
    print(f"Saved {bundle['metadata']['model_version']} ({bundle['metadata']['training_rows']:,} rows) to {ARTIFACT}")
