# AAAI2023 Dataset Naming Migration

## Goal

Make `aaai2023` the canonical dataset name while retaining `peiyou` as a
backward-compatible alias. New training runs, configuration references, and
user-facing labels use `aaai2023`; existing commands and historical artifact
paths containing `peiyou` remain valid.

## Naming and paths

- Rename the primary data directory from `data/peiyou` to `data/aaai2023`.
- Add `aaai2023` as the canonical key in `configs/data_config.json`.
- Keep a `peiyou` configuration alias pointing to the same canonical data
  directory.
- Add `configs/dataset/aaai2023.yaml`; retain `peiyou.yaml` as a compatibility
  configuration pointing to `data/aaai2023`.
- Normalize `peiyou` to `aaai2023` at the training boundary so new run names
  and saved configuration use the canonical name.

## Hidden-test discipline

Both names must retain the existing hidden-label behavior: do not calculate
test metrics from `-1` targets and generate predictions only. Error and warning
messages should use `AAAI2023` as the public display name.

## Prediction entry points

Add `scripts/predict_aaai2023.py` as the canonical prediction entry point.
Keep `scripts/predict_peiyou.py` operational for backward compatibility, but
change its defaults to the canonical dataset name and path. Shared behavior
must not be duplicated unnecessarily.

## Historical artifacts and documents

Do not rename existing `saved_model/cv-peiyou-*` directories or literal paths
that document historical artifacts. Those paths must continue to resolve.
User-facing dataset labels in active analysis code should become `AAAI2023`,
while compatibility logic may still recognize `peiyou`.

## Verification

- Parse JSON and YAML configurations.
- Verify both `aaai2023` and `peiyou` resolve to the same data files.
- Verify training normalization produces canonical `aaai2023` run names.
- Verify hidden-label test evaluation remains disabled for both input names.
- Run focused tests and syntax checks without changing checkpoints or data
  contents.
