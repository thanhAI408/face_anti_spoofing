# Evaluation workspace

`scripts/evaluate_texture.py` evaluates the passive texture model on the immutable test split at both frame and source-video levels. The threshold is selected only on validation data.

`manual_challenge_cases.csv` is the evidence sheet for end-to-end webcam tests. Fill `actual`, `duration_seconds`, `notes` and `evidence_path`; do not change expected results after testing.

Generated reports belong in `reports/evaluation/` and should be committed only for milestone releases together with the matching manifest/model hashes.
