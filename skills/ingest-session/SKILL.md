---
name: ingest-session
description: Validate and register a new vitiligo photo session (visible + UV images + metadata.json) against Imaging Protocol v1. Use when the user adds a new session folder under data/sessions/.
---

# ingest-session

1. Ask for (or locate) the session folder: `data/sessions/<patient-id>/<YYYY-MM-DD>/`.
2. If `metadata.json` is missing, create it interactively from `vititrack/ingest/schema.py` fields.
3. Run `python -c "from vititrack.ingest.validate import validate_session; import sys; m,p=validate_session(sys.argv[1]); print(p)" <folder>`.
4. Report protocol problems plainly (missing regions, missing UV shots, no colour card). Do not proceed to scoring until the session is compliant or the user explicitly overrides.
5. Strip EXIF GPS from images in place (`scripts/strip_exif.py`).
6. Append a line to `data/sessions/<patient-id>/index.csv`: date, regions, compliant (Y/N).

Never move, copy, or upload images outside `data/sessions/`.
