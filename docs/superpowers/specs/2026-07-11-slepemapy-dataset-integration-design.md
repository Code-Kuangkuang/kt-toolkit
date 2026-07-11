# Slepemapy Dataset Integration Design

## Objective

Connect the raw Slepemapy files under `data/slepemapy/` to KT-Toolkit's
existing cleaning entry point so the dataset can be preprocessed with the same
concept-level and question-level pipeline used by the other registered
datasets.

This change establishes and tests the integration path. It does not run the
full 10,087,305-interaction preprocessing job in this iteration.

## Confirmed Data Contract

The existing `preprocess/slepemapy_preprocess.py` defines the canonical mapping:

- student: `user`
- question: `(place_asked, type)`, serialized as `place_asked----type`
- knowledge concept: `place_asked`
- response: `1` when `place_asked == place_answered`, otherwise `0`
- event order: `inserted`, with the original row index as the stable tie-breaker
- use time: `response_time`

The local raw file contains 91,331 students, 2,913 questions, 1,458 concepts,
2,913 question-concept associations, and 10,087,305 interactions. No row is
removed by the existing required-field filter on `user`, `place_asked`, and
`inserted`.

## Selected Approach

Use the repository's standard dataset-adapter pattern without duplicating the
preprocessing or training framework:

1. Add a Slepemapy cleaning adapter that calls `process_raw_data`,
   `split_datasets.main`, and, when enabled, `split_datasets_que.main`.
2. Register the adapter in `cleaning.adapters.ADAPTERS`.
3. Add `configs/dataset/slepemapy.yaml` with the raw file path, five-fold split,
   sequence length 200, and question-level generation enabled.
4. Correct the Slepemapy `dpath` in `configs/data_config.json` to the repository
   local directory and declare the expected question-level file names and
   `max_concepts=1`.
5. Add a small end-to-end fixture test that exercises the real adapter and
   verifies both concept-level and question-level outputs.

This preserves the existing deterministic user-level split behavior
(`random_state=1024`) and keeps model/dataset loading behavior consistent with
the rest of KT-Toolkit.

## Files and Responsibilities

- `cleaning/adapters/slepemapy.py`: orchestrate the existing raw conversion and
  split functions for this dataset.
- `cleaning/adapters/__init__.py`: expose and register the new adapter.
- `configs/dataset/slepemapy.yaml`: provide the standard `run_clean.py`
  configuration.
- `configs/data_config.json`: correct the runtime data path and expected dataset
  metadata/file keys.
- `tests/test_slepemapy_integration.py`: verify registration, mapping, ordering,
  deterministic folds, and generated question-level files using temporary
  miniature data.

## Data Flow

```text
data/slepemapy/answer.csv
  -> preprocess.slepemapy_preprocess.read_data_from_csv
  -> data/slepemapy/data.txt
  -> preprocess.split_datasets.main
  -> concept-level train/valid/test artifacts
  -> preprocess.split_datasets_que.main
  -> question-level train/valid/test artifacts used by removed_model
```

`place.csv` and `place_type.csv` remain source metadata and are not used by the
current canonical mapping. Treating place type as an additional concept would
be a different modeling decision and is outside this integration.

## Error Handling and Safety

- The adapter relies on the existing preprocessing exceptions for missing or
  malformed input instead of swallowing failures.
- The YAML uses forward-slash paths because `process_raw_data` currently derives
  the output directory by splitting the raw path on `/`.
- Tests write only to a temporary directory and use a temporary data-config
  file, so they cannot overwrite the real dataset or shared configuration.
- The full raw preprocessing is intentionally deferred. The current splitter
  also creates test window artifacts, which can require substantial memory and
  disk space for a dataset of this size.
- Existing raw data and unrelated removed_model worktree changes are preserved and are
  not staged or modified.

## Test Strategy

The implementation follows red-green-refactor:

1. Add a test that initially fails because `slepemapy` is absent from
   `ADAPTERS`.
2. Add a miniature semicolon-delimited `answer.csv` covering correct, incorrect,
   and missing `place_answered` values, plus out-of-order timestamps.
3. Run the real cleaning adapter against the fixture.
4. Assert that responses are binary and correctly aligned after sorting.
5. Assert that train/test user IDs are disjoint, train folds are `0..4`, and
   test folds are `-1`.
6. Assert that concept-level and question-level core files exist and that the
   generated config reports `num_q`, `num_c`, and `max_concepts=1` consistently
   with the fixture.
7. Parse the repository YAML and JSON configuration and run Python syntax and
   adapter-registration checks.

## Non-Goals

- Running the complete 10-million-interaction conversion or training job.
- Redesigning shared split/window generation for large datasets.
- Changing the established Slepemapy question, concept, response, or time
  semantics.
- Adding place hierarchy or place type as model features.
- Modifying or committing raw data files.

## Acceptance Criteria

- `python scripts/run_clean.py --config configs/dataset/slepemapy.yaml` resolves
  to the existing raw file and dispatches through the registered adapter.
- The small end-to-end fixture test passes and produces loadable concept-level
  and question-level artifacts.
- `configs/data_config.json` resolves Slepemapy to `data/slepemapy` and declares
  `num_q=2913`, `num_c=1458`, and `max_concepts=1`.
- No unrelated file is modified or staged.
- The final handoff explicitly states that full preprocessing and long training
  have not been run.
