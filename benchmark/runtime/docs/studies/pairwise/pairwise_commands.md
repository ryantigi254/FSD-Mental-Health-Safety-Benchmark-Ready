# Pairwise Commands

## Write the canonical judge panel and default slice configs

```bash
python3 scripts/pairwise/write_default_pairwise_manifests.py
```

## Build one frozen case manifest

```bash
python3 scripts/pairwise/build_case_manifest.py --slice-id study_a
```

## Build all currently supported core manifests

```bash
for slice in study_a study_a_bias study_b study_b_multiturn study_c; do
  python3 scripts/pairwise/build_case_manifest.py --slice-id "$slice"
done
```

## Dry-run one slice

```bash
python3 scripts/pairwise/run_pairwise.py \
  --config metric-results/pairwise/manifests/configs/study_a.run_config.json \
  --dry-run
```

## Run one slice against LM Studio

```bash
python3 scripts/pairwise/run_pairwise.py \
  --config metric-results/pairwise/manifests/configs/study_a.run_config.json
```

## Generate the dedicated notebook family

```bash
python3 scripts/pairwise/generate_pairwise_notebooks.py
```
