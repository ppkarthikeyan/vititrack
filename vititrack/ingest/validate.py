"""Validate a session folder against Imaging Protocol v1."""
from __future__ import annotations
import json
from pathlib import Path
from .schema import SessionMetadata

def validate_session(folder: Path) -> tuple[SessionMetadata, list[str]]:
    """Return (metadata, problems). Empty problems == protocol-compliant."""
    folder = Path(folder)
    problems: list[str] = []
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing")
    meta = SessionMetadata(**json.loads(meta_path.read_text()))
    for region in meta.regions:
        vis = folder / f"{region}_vis.jpg"
        uv = folder / f"{region}_uv.jpg"
        if not vis.exists():
            problems.append(f"missing visible image for {region}")
        if meta.uv_captured and not uv.exists():
            problems.append(f"missing UV image for {region}")
    if not meta.colour_card_present:
        problems.append("colour card not present — colour normalisation will be skipped")
    return meta, problems
