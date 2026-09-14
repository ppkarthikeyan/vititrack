"""Download public vitiligo datasets listed in docs/DATA_SOURCES.md into data/public/.

Roboflow sets need a free API key: export ROBOFLOW_API_KEY=...  (app.roboflow.com → Settings → Roboflow Keys → Private)
Usage: python scripts/download_public_data.py
Already-downloaded sets are skipped; failures are reported but don't stop the run.
"""
import os, sys
from pathlib import Path

# (workspace, project, format)  — segmentation sets only; detection-only sets are not useful for Dice.
ROBOFLOW_PROJECTS = [
    ("vitiligo-rzgah", "viti-main", "coco-segmentation"),
    ("sedki", "vitiligo-amw4r", "coco-segmentation"),
    ("vitiligo-ubfrp", "vitiligo-o8hjs", "coco-segmentation"),
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
    ok, failed = [], []
    for ws, proj, fmt in ROBOFLOW_PROJECTS:
        dest = out / f"roboflow_{proj}"
        if dest.exists() and any(dest.iterdir()):
            print("already present, skipping", dest); ok.append(proj); continue
        try:
            p = rf.workspace(ws).project(proj)
            v = p.versions()[0]
            v.download(fmt, location=str(dest))
            print("downloaded", ws, proj); ok.append(proj)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {ws}/{proj}: {str(e)[:200]}"); failed.append(proj)
    print(f"\ndone. ok={ok} failed={failed}")

if __name__ == "__main__":
    main()
