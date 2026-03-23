# v5 invariance manifests

This directory stores deterministic sampling manifests generated from `data/frozen_splits/v5` for paired invariance runs.

Recommended commands:

- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_a --sample-size 150`
- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_b --sample-size 160`
- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_b_multi_turn --sample-size 12`
- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_c --sample-size 15`

To rebuild the full sampled split root in one step:

- `PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py`

Materialised generation inputs written here:

- `study_a_test.json`
- `study_b_test.json`
- `study_b_multi_turn_test.json`
- `study_c_test.json`
- `study_a/`
- `study_c/`
