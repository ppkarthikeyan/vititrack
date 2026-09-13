import json
from pathlib import Path
import numpy as np
import cv2
from vititrack.scoring.vasi import region_score, total_vasi, HAND_UNITS
from vititrack.scoring.segment import segment
from vititrack.scoring.run import score_session

def _synthetic_skin(w=400, h=300, patch=(150, 100, 250, 200), tone=(120, 160, 210)):
    """Brown-ish skin (BGR) with a pale rectangular patch."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = tone
    x0, y0, x1, y1 = patch
    img[y0:y1, x0:x1] = (235, 240, 245)
    return img

def test_region_score_math():
    lesion = np.zeros((10, 10), bool); lesion[:5] = True
    skin = np.ones((10, 10), bool)
    s = region_score("face", lesion, skin)
    assert abs(s.depig_fraction - 0.5) < 1e-9
    assert abs(s.vasi_contribution - HAND_UNITS["face"] * 0.5) < 1e-9
    assert abs(total_vasi([s, s]) - HAND_UNITS["face"]) < 1e-9

def test_colour_segmentation_finds_pale_patch():
    img = _synthetic_skin()
    lesion, skin = segment(img, "colour")
    assert skin.mean() > 0.9
    # the patch is 100x100 of 400x300 -> 1/12 of the image
    frac = lesion.sum() / skin.sum()
    assert 0.05 < frac < 0.12

def test_score_session_end_to_end(tmp_path: Path):
    meta = {"patient_id": "demo", "session_date": "2026-09-13", "phone_model": "test",
            "regions": ["face", "forearm_l"], "uv_captured": False}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    cv2.imwrite(str(tmp_path / "face_vis.jpg"), _synthetic_skin())
    cv2.imwrite(str(tmp_path / "forearm_l_vis.jpg"), _synthetic_skin(patch=(0, 0, 1, 1)))
    r = score_session(tmp_path)
    assert r["total_vasi"] > 0
    assert (tmp_path / "scoring" / "vasi.json").exists()
    assert (tmp_path / "scoring" / "face_overlay.jpg").exists()
    face = next(x for x in r["regions"] if x["region"] == "face")
    arm = next(x for x in r["regions"] if x["region"] == "forearm_l")
    assert face["depig_fraction"] > arm["depig_fraction"]
