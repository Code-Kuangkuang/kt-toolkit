# AAAI2023 Dataset Alias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `aaai2023` the canonical dataset identifier while preserving `peiyou` as a working compatibility alias and retaining hidden-test safeguards.

**Architecture:** Centralize dataset-name normalization in `core/dataset_names.py`, use it at CLI and training boundaries, and keep both configuration keys mapped to the canonical data directory. Move only the live data directory; preserve historical artifact paths containing `peiyou`.

**Tech Stack:** Python 3.12, Typer, JSON/YAML configuration, `unittest`, PyTorch dataset/training infrastructure.

## Global Constraints

- `aaai2023` is canonical; `peiyou` remains accepted as an alias.
- New run names and saved configurations use `aaai2023`.
- Both names disable metric evaluation on hidden `-1` test labels.
- Existing `saved_model/cv-peiyou-*` and documented historical paths are not renamed.
- No dataset contents, checkpoints, or response labels are modified.

---

### Task 1: Canonical dataset-name resolver and training behavior

**Files:**
- Create: `core/dataset_names.py`
- Modify: `scripts/train.py`
- Modify: `core/train_runner.py`
- Test: `tests/test_aaai2023_alias.py`

**Interfaces:**
- Produces: `normalize_dataset_name(name: str) -> str` and `is_hidden_label_dataset(name: str) -> bool`.
- Consumes: raw CLI dataset names and passes canonical names into configuration lookup, run naming, and hidden-test selection.

- [ ] **Step 1: Write failing resolver and runner tests**

```python
def test_peiyou_normalizes_to_aaai2023():
    from core.dataset_names import normalize_dataset_name
    assert normalize_dataset_name("peiyou") == "aaai2023"
    assert normalize_dataset_name("AAAI2023") == "aaai2023"

def test_both_names_are_hidden_label_datasets():
    from core.dataset_names import is_hidden_label_dataset
    assert is_hidden_label_dataset("peiyou")
    assert is_hidden_label_dataset("aaai2023")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `C:\Users\10577\.conda\envs\pykt312\python.exe -m unittest tests.test_aaai2023_alias -v`

Expected: FAIL because `core.dataset_names` does not exist.

- [ ] **Step 3: Implement the resolver**

```python
DATASET_NAME_ALIASES = {"peiyou": "aaai2023"}
HIDDEN_LABEL_DATASETS = {"aaai2023"}

def normalize_dataset_name(name):
    normalized = str(name).strip().lower()
    return DATASET_NAME_ALIASES.get(normalized, normalized)

def is_hidden_label_dataset(name):
    return normalize_dataset_name(name) in HIDDEN_LABEL_DATASETS
```

Call `normalize_dataset_name()` at the start of `launch_train()`, `main()` before CV directory naming, and `train_one_fold()`. Replace `is_peiyou` with `is_hidden_label_dataset(dataset_name)` and update the message to point to `scripts/predict_aaai2023.py`.

- [ ] **Step 4: Run focused tests**

Run: `C:\Users\10577\.conda\envs\pykt312\python.exe -m unittest tests.test_aaai2023_alias -v`

Expected: resolver and hidden-label tests PASS.

---

### Task 2: Canonical configuration and live data path

**Files:**
- Modify: `configs/data_config.json`
- Create: `configs/dataset/aaai2023.yaml`
- Modify: `configs/dataset/peiyou.yaml`
- Modify: `preprocess/data_proprocess.py`
- Move: `data/peiyou` to `data/aaai2023`
- Test: `tests/test_aaai2023_alias.py`

**Interfaces:**
- Consumes: canonical name from Task 1.
- Produces: both config keys resolving to `data/aaai2023`, with identical counts, folds, and sequence filenames.

- [ ] **Step 1: Add failing configuration tests**

```python
def test_both_config_names_share_canonical_path():
    config = json.loads((ROOT / "configs/data_config.json").read_text())
    assert config["aaai2023"]["dpath"] == "data/aaai2023"
    assert config["peiyou"] == config["aaai2023"]
    assert (ROOT / config["aaai2023"]["dpath"]).is_dir()
```

- [ ] **Step 2: Run test and verify RED**

Run the focused unittest command. Expected: FAIL because `aaai2023` is absent.

- [ ] **Step 3: Update configuration and preprocessing aliases**

Duplicate the existing dataset object under canonical key `aaai2023`, set both objects' `dpath` to `data/aaai2023`, and keep their remaining fields identical. Create canonical YAML with `dataset_name: aaai2023`; retain compatibility YAML with `dataset_name: peiyou`; both use `dpath: data/aaai2023`. Change preprocessing branches to `dataset_name in {"aaai2023", "peiyou"}`.

- [ ] **Step 4: Move the directory safely**

Resolve `data/peiyou` and `data/aaai2023`; require the source to be inside the repository and destination not to exist, then use PowerShell `Move-Item -LiteralPath ... -Destination ...`.

- [ ] **Step 5: Verify configuration and paths**

Run: `python -m unittest tests.test_aaai2023_alias -v`

Run: `python -c "import json; json.load(open('configs/data_config.json', encoding='utf-8')); print('valid')"`

Expected: tests PASS and JSON prints `valid`.

---

### Task 3: Canonical prediction entry point with legacy wrapper

**Files:**
- Move: `scripts/predict_peiyou.py` to `scripts/predict_aaai2023.py`
- Create: `scripts/predict_peiyou.py`
- Test: `tests/test_aaai2023_alias.py`

**Interfaces:**
- Produces: canonical Typer `app` in `scripts.predict_aaai2023`; legacy script imports and executes the same app.
- Consumes: canonical `aaai2023` config and `data/aaai2023/pykt_test.csv`.

- [ ] **Step 1: Add failing entry-point tests**

```python
def test_prediction_entry_points_share_one_app():
    from scripts import predict_aaai2023, predict_peiyou
    assert predict_peiyou.app is predict_aaai2023.app
```

- [ ] **Step 2: Run test and verify RED**

Run the focused unittest command. Expected: FAIL because `predict_aaai2023.py` does not exist.

- [ ] **Step 3: Move implementation and add compatibility wrapper**

Change canonical defaults to `ROOT / "data" / "aaai2023" / "pykt_test.csv"` and `dataset_name="aaai2023"`. The compatibility file contains:

```python
from scripts.predict_aaai2023 import app, main

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run focused and integration verification**

Run the focused unittest command, `python -m py_compile` on changed Python files, registry import checks, and `git diff --check`.

Expected: all commands exit 0; both entry points expose the same app; no hidden test metric loader is created for either dataset name.

---

### Task 4: Final scope and compatibility audit

**Files:**
- Review only: repository references and Git diff.

**Interfaces:**
- Consumes all deliverables from Tasks 1–3.
- Produces a verified migration report with unmodified historical paths explicitly listed as intentional.

- [ ] **Step 1: Search active references**

Run `rg -n -i "peiyou" core scripts configs preprocess tests --glob '!scripts/predict_peiyou.py'` and classify every remaining occurrence as compatibility logic or historical reference.

- [ ] **Step 2: Verify no unrelated changes**

Run `git status --short` and `git diff -- core/dataset_names.py scripts/train.py core/train_runner.py configs/data_config.json configs/dataset/aaai2023.yaml configs/dataset/peiyou.yaml preprocess/data_proprocess.py scripts/predict_aaai2023.py scripts/predict_peiyou.py tests/test_aaai2023_alias.py`.

- [ ] **Step 3: Run final tests**

Run the focused test suite plus the existing DIMKT leakage regression test. Expected: all tests PASS.
