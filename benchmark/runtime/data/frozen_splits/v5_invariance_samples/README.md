# v5 invariance manifests

This directory stores deterministic sampling manifests generated from `data/frozen_splits/v5` for paired invariance runs.

Recommended commands:

- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_a --sample-size 150`
- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_b --sample-size 160`
- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_b_multi_turn --sample-size 12`
- `PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_c --sample-size 15`
