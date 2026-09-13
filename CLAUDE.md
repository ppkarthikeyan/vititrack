# VitiTrack — notes for Claude Code

- Research/decision-support project for early-stage vitiligo. Never give dosing advice or diagnoses in outputs; cite guidelines instead.
- Patient data lives only in `data/sessions/` and is gitignored. Never commit, upload, or copy it elsewhere.
- Always report model metrics stratified by Fitzpatrick skin type.
- Skills live in `skills/<name>/SKILL.md`; build order is in README.
- Run `pytest` before committing.
