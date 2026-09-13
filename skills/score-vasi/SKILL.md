---
name: score-vasi
description: Segment depigmented skin in a validated session and compute VASI per body region and in total. Use after ingest-session, or when the user asks for a VASI score, extent, or overlay images.
---

# score-vasi

1. Confirm the session folder passed `ingest-session` (no blocking protocol problems).
2. Run: `python -m vititrack.scoring.run data/sessions/<patient-id>/<YYYY-MM-DD>`
3. Read `scoring/vasi.json`. Report: total VASI, the top 3 contributing regions, and the backend used.
4. Show the user the overlay images (`scoring/<region>_overlay.jpg`) and ask whether the red mask looks right. If it clearly over- or under-segments, suggest re-shooting per `docs/IMAGING_PROTOCOL.md` (lighting is the usual cause) or, for the colour backend, adjusting `delta_L`.
5. Append the total to `data/sessions/<patient-id>/vasi_history.csv` (date, total_vasi, backend).

Notes
- The `colour` backend is a rough baseline. Say so. Numbers are for trend tracking, not for comparison with a clinician's VASI until the learned backend exists.
- `lesion_site` close-ups are not scored (0 hand units); they are for change detection.
- Never move images outside `data/sessions/`.
