# Imaging Protocol v1

Goal: repeatable monthly photos so that session *t* can be registered to session *t−1*.

## Kit
- Smartphone (same phone every time), small tripod
- 365 nm UV torch (LED, ≥3 W)
- Printed colour/grey card (e.g. an 18% grey card + 6-patch colour strip), laminated
- Floor tape marks for standing position; tripod height noted

## Per session
For each region — face, neck, hands (dorsal/palmar), forearms, upper arms, trunk front/back, thighs, shins, feet — plus any individual lesion sites:
1. **Visible**: diffuse daylight or room light, flash OFF, colour card in frame at the same edge, region fills ~70% of the frame.
2. **UV**: same framing, room dark, UV torch ~30 cm from skin, held perpendicular. Vitiligo fluoresces bright blue-white. Ill-defined / feathery edges suggest activity.
3. Save as `data/sessions/<patient-id>/<YYYY-MM-DD>/<region>_<vis|uv>.jpg`
4. Fill `metadata.json` (schema in `vititrack/ingest/schema.py`): date, phone model, lighting, regions captured, treatment log since last session, patient-reported new spots (Y/N), itch (Y/N), trauma/friction sites.

## Frequency
Monthly. Every 2 weeks if disease is active or treatment was just changed.

## Don'ts
- No filters, no HDR, no portrait mode.
- No zoom — move the tripod instead.
- Don't photograph under mixed light sources.
