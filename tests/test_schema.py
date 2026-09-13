import json
from pathlib import Path
from vititrack.ingest.validate import validate_session

def test_validate_session(tmp_path: Path):
    meta = {
        "patient_id": "demo", "session_date": "2026-09-06", "phone_model": "test",
        "regions": ["face", "hand_dorsal_l"], "uv_captured": True,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    for r in meta["regions"]:
        (tmp_path / f"{r}_vis.jpg").write_bytes(b"")
    (tmp_path / "face_uv.jpg").write_bytes(b"")
    m, problems = validate_session(tmp_path)
    assert m.patient_id == "demo"
    assert problems == ["missing UV image for hand_dorsal_l"]
