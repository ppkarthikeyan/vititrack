# VitiTrack

**Open, agent-driven longitudinal monitoring for early-stage vitiligo.**

VitiTrack turns serial phone photos (visible light + 365 nm UV) into a longitudinal record of lesion **extent** (VASI) and disease **activity** (is it spreading?), using the patient as their own control. It adds guideline-referenced decision support and clinical-trial matching to help patients and dermatologists act in the window when treatment works best.

> ⚠️ Research software. Decision-support only — not a medical device, not a diagnosis, never a prescription. All patient data stays on the user's machine.

## Why

- Every 2025–26 review of AI in vitiligo names the same gaps: small private datasets, no standard imaging protocol, poor coverage of darker skin (Fitzpatrick IV–VI), and no tool that measures *activity* rather than just area.
- Early active disease is the therapeutic window. Knowing whether lesions are spreading — and whether treatment is working — is what changes management.
- Serial change detection on one person needs far less data than population classifiers.

## What's in the box

```
vititrack/          Python package
  ingest/           protocol validation, metadata schema, colour-card normalisation
  scoring/          segmentation → VASI per body region
  change/           registration + UV-mask differencing across sessions (ΔVASI, new lesions)
  activity/         confetti / Koebner / border-blur detector → activity score 0–3
  report/           one-page trend report for the dermatologist
skills/             Claude Code skills that orchestrate the package (see below)
data/
  public/           downloaded public datasets (gitignored images; manifests only)
  eval/             Fitzpatrick-stratified evaluation set manifests + masks
  sessions/         patient sessions — NEVER committed
docs/               imaging protocol, consent template, data policy, model cards
notebooks/          experiments
tests/
```

## Agent skills

| Skill | Input → Output |
|---|---|
| `ingest-session` | photo folder + metadata → validated, region-tagged session |
| `score-vasi` | session → masks, VASI per region, total |
| `detect-change` | session t, t−1 → ΔVASI, new / grown / repigmented lesions |
| `assess-activity` | session → activity score + evidence crops |
| `weekly-report` | everything → one-page report |
| `guideline-options` | record → guideline-cited options for discussion with the doctor |
| `match-trials` | record + country → ClinicalTrials.gov / CTRI matches |
| `pipeline-watch` | scheduled: trial readouts, drug availability, phototherapy centres |

Build order: `ingest-session` → `score-vasi` → `weekly-report` → `detect-change` → `assess-activity` → the rest.

## Quick start

```bash
git clone https://github.com/<you>/vititrack.git
cd vititrack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data

No patient images are in this repo. Public sources used for baselines and the evaluation set are listed in `docs/DATA_SOURCES.md` with licences. Longitudinal validation uses the VR Foundation CloudBank (academic access) and partner-clinic data under ethics approval.

## Roadmap

See `docs/ROADMAP.md`. Milestones: protocol v1 → stratified eval set + baselines → end-to-end harness → change detection + activity score → open benchmark release → response-prediction on CloudBank data.

## Contributing

Issues and PRs welcome, especially: images/labels under open licences, dermatologist review of the protocol, and skin-tone-diverse evaluation data. See `CONTRIBUTING.md`.

## Citation

If you use VitiTrack, please cite this repository (a preprint is planned).

## License

Apache-2.0 for code. Data licences are per-source (see `docs/DATA_SOURCES.md`).
# vititrack
