"""Download public vitiligo datasets listed in docs/DATA_SOURCES.md into data/public/.

Roboflow sets need a free API key: export ROBOFLOW_API_KEY=...  (roboflow.com → Settings → API)
Usage: python scripts/download_public_data.py
"""
import os, sys
from pathlib import Path

ROBOFLOW_PROJECTS = [
    ("vitiligo-rzgah", "viti-main"),
    ("futureofikigai", "vitiligo_1"),
    ("sedki", "vitiligo-amw4r"),
]

def main():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY first (free at roboflow.com)")
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("pip install roboflow")
    rf = Roboflow(api_key=key)
    out = Path("data/public"); out.mkdir(parents=True, exist_ok=True)
    for ws, proj in ROBOFLOW_PROJECTS:
        p = rf.workspace(ws).project(proj)
        v = p.versions()[0]
        v.download("coco-segmentation", location=str(out / f"roboflow_{proj}"))
        print("downloaded", ws, proj)

if __name__ == "__main__":
    main()
