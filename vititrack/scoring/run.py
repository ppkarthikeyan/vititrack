"""Score a session folder: python -m vititrack.scoring.run data/sessions/<pid>/<date>"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np
from vititrack.ingest.validate import validate_session
from .segment import segment
from .vasi import region_score, total_vasi, RegionScore

def score_session(folder: Path, backend: str = "colour", save_overlays: bool = True) -> dict:
    folder = Path(folder)
    meta, problems = validate_session(folder)
    scores: list[RegionScore] = []
    out_dir = folder / "scoring"
    out_dir.mkdir(exist_ok=True)
    for region in meta.regions:
        img_path = folder / f"{region}_vis.jpg"
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        lesion, skin = segment(bgr, backend)
        s = region_score(region, lesion, skin)
        scores.append(s)
        np.save(out_dir / f"{region}_lesion.npy", lesion)
        if save_overlays:
            ov = bgr.copy()
            ov[lesion] = (0.4 * ov[lesion] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{region}_overlay.jpg"), ov)
    result = {
        "patient_id": meta.patient_id,
        "session_date": str(meta.session_date),
        "backend": backend,
        "protocol_problems": problems,
        "regions": [s.__dict__ for s in scores],
        "total_vasi": round(total_vasi(scores), 2),
    }
    (out_dir / "vasi.json").write_text(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    r = score_session(Path(sys.argv[1]))
    print(json.dumps(r, indent=2))
