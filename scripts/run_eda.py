"""Execute notebooks/01_eda.ipynb headlessly to create the frozen event split and EDA artifacts.

The notebook writes artifacts/event_splits.csv, artifacts/eda_manifest.json and reports/eda_summary.md;
reruns verify and reuse an existing split. The notebook file itself is not modified.
"""
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    path=ROOT/'notebooks/01_eda.ipynb'
    notebook=nbformat.read(path,as_version=4)
    started=time.time()
    NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
    for required in ['artifacts/event_splits.csv','artifacts/eda_manifest.json']:
        if not (ROOT/required).exists():
            sys.exit(f'EDA finished but {required} is missing.')
    print(f'EDA notebook executed in {time.time()-started:.0f}s; event split ready.')
