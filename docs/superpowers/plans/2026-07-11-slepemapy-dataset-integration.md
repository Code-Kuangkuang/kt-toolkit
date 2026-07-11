# Slepemapy Dataset Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register Slepemapy with KT-Toolkit's cleaning entry point and prove with a temporary end-to-end fixture that the existing preprocessing pipeline generates aligned concept-level and question-level artifacts.

**Architecture:** Add the same thin cleaning adapter and dataset YAML used by the repository's other datasets, while leaving the established Slepemapy mapping and shared splitters unchanged. A standard-library `unittest` test will run the real adapter in a temporary directory, so the integration is verified without touching the 702 MB raw file or the shared data configuration.

**Tech Stack:** Python 3.12, pandas 2.3.3, PyYAML 6.0.3, `unittest`, existing KT-Toolkit preprocessing modules.

## Global Constraints

- Do not run the complete 10,087,305-interaction preprocessing or a training job.
- Preserve the canonical mapping: student=`user`, question=`place_asked----type`, concept=`place_asked`, response=`place_asked == place_answered`.
- Preserve chronological ordering by `inserted`, with original row index as the stable tie-breaker.
- Do not redesign `preprocess/split_datasets.py` or `preprocess/split_datasets_que.py`.
- Do not add place hierarchy or place type as model features.
- Do not modify, stage, or commit files under `data/slepemapy/`.
- Preserve all unrelated removed_model worktree changes.
- Use `C:\Users\10577\.conda\envs\pykt312\python.exe` for tests because it provides pandas and PyYAML; no new dependency installation is required.

---

### Task 1: Add a failing end-to-end integration contract

**Files:**
- Create: `tests/test_slepemapy_integration.py`

**Interfaces:**
- Consumes: `cleaning.adapters.ADAPTERS`, repository YAML/JSON config files, and the existing `run(cfg: dict)` adapter convention.
- Produces: two executable `unittest` cases describing registration/config behavior and the raw-to-question-level artifact contract.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slepemapy_integration.py` with this complete content:

```python
import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cleaning.adapters import ADAPTERS


ROOT = Path(__file__).resolve().parents[1]


class SlepemapyIntegrationTest(unittest.TestCase):
    def test_repository_config_and_adapter_registration(self):
        self.assertIn("slepemapy", ADAPTERS)

        yaml_path = ROOT / "configs" / "dataset" / "slepemapy.yaml"
        with yaml_path.open("r", encoding="utf-8") as fin:
            dataset_yaml = yaml.safe_load(fin)
        self.assertEqual(dataset_yaml["dataset_name"], "slepemapy")
        self.assertEqual(dataset_yaml["raw_path"], "data/slepemapy/answer.csv")
        self.assertEqual(dataset_yaml["dpath"], "data/slepemapy")
        self.assertTrue(dataset_yaml["gen_question_level"])

        with (ROOT / "configs" / "data_config.json").open("r", encoding="utf-8") as fin:
            dataset_cfg = json.load(fin)["slepemapy"]
        expected = {
            "dpath": "data/slepemapy",
            "num_q": 2913,
            "num_c": 1458,
            "max_concepts": 1,
            "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
            "test_file_quelevel": "test_sequences_quelevel.csv",
        }
        for key, value in expected.items():
            self.assertEqual(dataset_cfg[key], value)

    def test_adapter_generates_aligned_question_level_artifacts(self):
        self.assertIn("slepemapy", ADAPTERS)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_path = tmp_path / "answer.csv"
            config_path = tmp_path / "data_config.json"
            config_path.write_text("{}", encoding="utf-8")

            fieldnames = [
                "id",
                "user",
                "place_asked",
                "place_answered",
                "type",
                "inserted",
                "response_time",
            ]
            rows = []
            answer_id = 1
            for offset in range(10):
                user = 1000 + offset
                rows.extend(
                    [
                        {
                            "id": answer_id,
                            "user": user,
                            "place_asked": 2,
                            "place_answered": 1,
                            "type": 2,
                            "inserted": "2015-01-01 00:00:02",
                            "response_time": 2000,
                        },
                        {
                            "id": answer_id + 1,
                            "user": user,
                            "place_asked": 1,
                            "place_answered": 1,
                            "type": 1,
                            "inserted": "2015-01-01 00:00:01",
                            "response_time": 1000,
                        },
                        {
                            "id": answer_id + 2,
                            "user": user,
                            "place_asked": 3,
                            "place_answered": "",
                            "type": 1,
                            "inserted": "2015-01-01 00:00:03",
                            "response_time": 3000,
                        },
                    ]
                )
                answer_id += 3

            with raw_path.open("w", encoding="utf-8", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)

            cfg = {
                "dataset_name": "slepemapy",
                "raw_path": raw_path.as_posix(),
                "dpath": tmp_path.as_posix(),
                "configf": config_path.as_posix(),
                "min_seq_len": 3,
                "maxlen": 8,
                "kfold": 5,
                "gen_question_level": True,
            }
            ADAPTERS["slepemapy"](cfg)

            data_lines = (tmp_path / "data.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(data_lines[0], "1000,3")
            self.assertEqual(data_lines[1], "1----1,2----2,3----1")
            self.assertEqual(data_lines[2], "1,2,3")
            self.assertEqual(data_lines[3], "1,0,0")

            core_outputs = {
                "train_valid.csv",
                "train_valid_sequences.csv",
                "test.csv",
                "test_sequences.csv",
                "train_valid_quelevel.csv",
                "train_valid_sequences_quelevel.csv",
                "test_quelevel.csv",
                "test_sequences_quelevel.csv",
                "keyid2idx.json",
            }
            self.assertTrue(all((tmp_path / name).is_file() for name in core_outputs))

            with (tmp_path / "train_valid_quelevel.csv").open(
                "r", encoding="utf-8", newline=""
            ) as fin:
                train_rows = list(csv.DictReader(fin))
            with (tmp_path / "test_quelevel.csv").open(
                "r", encoding="utf-8", newline=""
            ) as fin:
                test_rows = list(csv.DictReader(fin))

            train_users = {row["uid"] for row in train_rows}
            test_users = {row["uid"] for row in test_rows}
            self.assertTrue(train_users.isdisjoint(test_users))
            self.assertEqual({int(row["fold"]) for row in train_rows}, set(range(5)))
            self.assertEqual({int(row["fold"]) for row in test_rows}, {-1})

            for row in train_rows + test_rows:
                lengths = {
                    len(row["questions"].split(",")),
                    len(row["concepts"].split(",")),
                    len(row["responses"].split(",")),
                }
                self.assertEqual(lengths, {3})
                self.assertLessEqual(set(row["responses"].split(",")), {"0", "1"})

            generated_cfg = json.loads(config_path.read_text(encoding="utf-8"))["slepemapy"]
            self.assertEqual(generated_cfg["num_q"], 3)
            self.assertEqual(generated_cfg["num_c"], 3)
            self.assertEqual(generated_cfg["max_concepts"], 1)
            self.assertEqual(
                generated_cfg["train_valid_file_quelevel"],
                "train_valid_sequences_quelevel.csv",
            )
            self.assertEqual(
                generated_cfg["test_file_quelevel"],
                "test_sequences_quelevel.csv",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the expected RED state**

Run:

```powershell
& 'C:\Users\10577\.conda\envs\pykt312\python.exe' -m unittest discover -s tests -p 'test_slepemapy_integration.py' -v
```

Expected: both tests report `FAIL` at `self.assertIn("slepemapy", ADAPTERS)` because the adapter is not registered. There must be no import, syntax, or fixture-construction error.

- [ ] **Step 3: Commit the verified failing contract**

```powershell
git add -- tests/test_slepemapy_integration.py
git commit -m "test: define Slepemapy integration contract"
```

---

### Task 2: Implement the standard Slepemapy adapter and configuration

**Files:**
- Create: `cleaning/adapters/slepemapy.py`
- Modify: `cleaning/adapters/__init__.py`
- Create: `configs/dataset/slepemapy.yaml`
- Modify: `configs/data_config.json`
- Test: `tests/test_slepemapy_integration.py`

**Interfaces:**
- Consumes: `process_raw_data(dataset_name, dname2paths)`, `split_datasets.main(...)`, `split_datasets_que.main(...)`, and the standard cleaning `cfg` mapping.
- Produces: `run(cfg)` and the `ADAPTERS["slepemapy"]` registration, plus runtime metadata that directs removed_model to explicit question-level files.

- [ ] **Step 1: Add the thin adapter**

Create `cleaning/adapters/slepemapy.py`:

```python
from preprocess.data_proprocess import process_raw_data
from preprocess.split_datasets import main as split_concept
from preprocess.split_datasets_que import main as split_question


def run(cfg):
    dname2paths = {"slepemapy": cfg["raw_path"]}
    dname, writef = process_raw_data(cfg["dataset_name"], dname2paths)

    split_concept(
        dname,
        writef,
        cfg["dataset_name"],
        cfg["configf"],
        cfg["min_seq_len"],
        cfg["maxlen"],
        cfg["kfold"],
    )
    if cfg.get("gen_question_level", True):
        split_question(
            dname,
            writef,
            cfg["dataset_name"],
            cfg["configf"],
            cfg["min_seq_len"],
            cfg["maxlen"],
            cfg["kfold"],
        )
```

- [ ] **Step 2: Register the adapter**

Add this import to `cleaning/adapters/__init__.py`:

```python
from .slepemapy import run as slepemapy
```

Add this entry to `ADAPTERS`:

```python
"slepemapy": slepemapy,
```

- [ ] **Step 3: Add the dataset YAML**

Create `configs/dataset/slepemapy.yaml`:

```yaml
dataset_name: slepemapy
raw_path: data/slepemapy/answer.csv
dpath: data/slepemapy
configf: configs/data_config.json

min_seq_len: 3
maxlen: 200
kfold: 5
folds: [0,1,2,3,4]
emb_path: ""

train_valid_original_file: train_valid.csv
train_valid_file: train_valid_sequences.csv
test_original_file: test.csv
test_file: test_sequences.csv
test_window_file: test_window_sequences.csv

gen_question_level: true
test_question_file: test_question_sequences.csv
test_question_window_file: test_question_window_sequences.csv
train_valid_original_file_quelevel: train_valid_quelevel.csv
train_valid_file_quelevel: train_valid_sequences_quelevel.csv
test_file_quelevel: test_sequences_quelevel.csv
test_window_file_quelevel: test_window_sequences_quelevel.csv
test_original_file_quelevel: test_quelevel.csv
```

- [ ] **Step 4: Correct and complete the runtime data config**

Replace the existing `slepemapy` object in `configs/data_config.json` with:

```json
"slepemapy": {
    "dpath": "data/slepemapy",
    "num_q": 2913,
    "num_c": 1458,
    "input_type": [
        "questions",
        "concepts"
    ],
    "max_concepts": 1,
    "min_seq_len": 3,
    "maxlen": 200,
    "emb_path": "",
    "train_valid_original_file": "train_valid.csv",
    "train_valid_file": "train_valid_sequences.csv",
    "folds": [
        0,
        1,
        2,
        3,
        4
    ],
    "test_original_file": "test.csv",
    "test_file": "test_sequences.csv",
    "test_window_file": "test_window_sequences.csv",
    "test_question_file": "test_question_sequences.csv",
    "test_question_window_file": "test_question_window_sequences.csv",
    "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
    "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
    "test_file_quelevel": "test_sequences_quelevel.csv",
    "test_window_file_quelevel": "test_window_sequences_quelevel.csv",
    "test_original_file_quelevel": "test_quelevel.csv"
}
```

- [ ] **Step 5: Run the tests and verify GREEN**

Run:

```powershell
& 'C:\Users\10577\.conda\envs\pykt312\python.exe' -m unittest discover -s tests -p 'test_slepemapy_integration.py' -v
```

Expected: `Ran 2 tests` followed by `OK`. The command may print existing preprocessing statistics, but it must emit no traceback and must not create files under the repository's `data/slepemapy/` directory.

- [ ] **Step 6: Commit the minimal implementation**

```powershell
git add -- cleaning/adapters/slepemapy.py cleaning/adapters/__init__.py configs/dataset/slepemapy.yaml configs/data_config.json
git commit -m "feat: integrate Slepemapy preprocessing"
```

---

### Task 3: Run final focused verification

**Files:**
- Verify: `tests/test_slepemapy_integration.py`
- Verify: `cleaning/adapters/slepemapy.py`
- Verify: `cleaning/adapters/__init__.py`
- Verify: `configs/dataset/slepemapy.yaml`
- Verify: `configs/data_config.json`

**Interfaces:**
- Consumes: the completed Task 1 contract and Task 2 adapter/configuration.
- Produces: fresh evidence that the integration is registered, syntactically valid, source configuration is parseable, and the real raw directory was not mutated.

- [ ] **Step 1: Re-run the complete targeted integration test**

```powershell
& 'C:\Users\10577\.conda\envs\pykt312\python.exe' -m unittest discover -s tests -p 'test_slepemapy_integration.py' -v
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 2: Verify syntax, registration, and configuration parsing**

```powershell
& 'C:\Users\10577\.conda\envs\pykt312\python.exe' -m py_compile cleaning/adapters/slepemapy.py tests/test_slepemapy_integration.py
& 'C:\Users\10577\.conda\envs\pykt312\python.exe' -c "import json, yaml; from cleaning.adapters import ADAPTERS; y=yaml.safe_load(open('configs/dataset/slepemapy.yaml', encoding='utf-8')); c=json.load(open('configs/data_config.json', encoding='utf-8'))['slepemapy']; assert 'slepemapy' in ADAPTERS; assert y['raw_path']=='data/slepemapy/answer.csv'; assert c['dpath']=='data/slepemapy' and c['num_q']==2913 and c['num_c']==1458 and c['max_concepts']==1; print('SLEPEMAPY_CONFIG_OK')"
```

Expected: both commands exit `0`, and the second prints `SLEPEMAPY_CONFIG_OK`.

- [ ] **Step 3: Verify the real raw directory was not changed by the tests**

```powershell
$expected = @('README.md','answer.csv','place.csv','place_type.csv')
$actual = Get-ChildItem data\slepemapy -File | Select-Object -ExpandProperty Name | Sort-Object
if (Compare-Object ($expected | Sort-Object) $actual) { throw 'Unexpected generated file in data/slepemapy' }
Write-Output 'RAW_DIRECTORY_UNCHANGED'
```

Expected: `RAW_DIRECTORY_UNCHANGED`.

- [ ] **Step 4: Inspect only the Slepemapy integration diff and repository status**

```powershell
git diff --check 0179d30..HEAD -- cleaning/adapters configs/dataset/slepemapy.yaml configs/data_config.json tests/test_slepemapy_integration.py
git status --short
```

Expected: the scoped `git diff --check` exits `0`; `git status --short` still shows the user's pre-existing removed_model changes and untracked raw Slepemapy directory, with no generated full preprocessing artifacts.

- [ ] **Step 5: Report the verified boundary**

The handoff must state that adapter registration, config parsing, and temporary end-to-end preprocessing passed, while full Slepemapy preprocessing and model training remain intentionally unexecuted.
