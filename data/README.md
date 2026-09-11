# Local biometric data (not committed)

Raw face videos and processed crops are intentionally ignored by Git.

Expected layout:

```text
raw/live/<subject_id>/<session_id>/<video>.mp4
raw/spoof/<attack_type>/<subject_id>/<session_id>/<video>.mp4
```

Run `python scripts/prepare_dataset.py` from the repository root. See `docs/DATASET_PROTOCOL.md` for consent, metadata, split and evaluation requirements.
