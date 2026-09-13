# Data handling policy

- Patient sessions live only in `data/sessions/` on the user's machine and are gitignored.
- Exports are anonymised (patient id → random token, EXIF stripped) and only produced on explicit user action.
- Any data from a partner clinic is used under that clinic's ethics approval and data-sharing agreement.
- Family/self data, if used at all, is a protocol test-bed only and never part of a published evaluation.
- Deletion on request: `python scripts/delete_patient.py <patient-id>`.
