"""Protocol-level leakage audit for both dataset modes.

The leak this guards against is not in any model: it is in the data pipeline,
so one check covers every registered model.  A scored position leaks when the
answer it is asked to predict is already sitting in the model's own input
history for the SAME question -- which is exactly what happened when
`is_repeat` was never read and the extra KC rows of a multi-KC question were
all scored.

The pass/fail criterion is narrow on purpose:

  FAIL   a scored position carries `is_repeat == 1`.  Those rows are one
         attempt split across its KCs, so the answer is copied, not predicted.

  REPORT the share of scored positions whose (question, response) pair the
         history already holds.  This is NOT a failure: a learner genuinely
         re-attempting a question they already answered the same way is real
         signal, and every deployment has it.  It is reported because the two
         dataset modes must agree on it -- a gap between them means one of them
         is still counting split attempts as re-attempts.

Run directly for a report over several datasets:

    python tests/test_no_target_leakage.py
    python tests/test_no_target_leakage.py assist2009 algebra2005
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

# Datasets whose question-level files exist and are cheap enough to scan.
DEFAULT_DATASETS = ["assist2009", "algebra2005", "bridge2algebra2006"]
MAX_ROWS = 400


def _seq(row, key):
    value = row.get(key, "")
    if value is None or value == "":
        return []
    return [int(float(x)) for x in str(value).split(",") if x != ""]


def audit_sequence_file(path, max_rows=MAX_ROWS):
    """Count scored positions whose label is already visible in their history.

    Returns a dict; `leaked` is the number of scored positions whose question
    appears earlier in the same sequence carrying the same response.  Under
    `one_by_one` those earlier rows are the other KCs of the same question.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=max_rows)
    has_repeat_col = "is_repeat" in df.columns
    scored = leaked = repeat_scored = 0

    for _, row in df.iterrows():
        responses = _seq(row, "responses")
        if not responses:
            continue
        questions = _seq(row, "questions") or _seq(row, "concepts")
        masks = _seq(row, "selectmasks") or [1 if r != -1 else -1 for r in responses]
        repeats = _seq(row, "is_repeat") if has_repeat_col else []

        seen = {}  # question id -> responses already fed as input
        for i, response in enumerate(responses):
            if response == -1 or i >= len(questions):
                continue
            question = questions[i]
            is_scored = i < len(masks) and masks[i] == 1
            if is_scored:
                scored += 1
                if response in seen.get(question, ()):
                    leaked += 1
                if repeats and i < len(repeats) and repeats[i] == 1:
                    repeat_scored += 1
            # Position i becomes history for every later position.
            seen.setdefault(question, set()).add(response)

    return {
        "path": Path(path).name,
        "scored": scored,
        "leaked": leaked,
        "repeat_scored": repeat_scored,
        "has_repeat_col": has_repeat_col,
    }


def _apply_repository_mask(path, max_rows=MAX_ROWS):
    """Audit using the mask the repository actually builds, so the result
    follows `KT_SCORE_REPEATED_KC` instead of trusting the raw column.

    `split_attempt` is the hard failure: scored rows that are the same attempt
    expanded across its KCs.  `reattempt` is the diagnostic: scored positions
    whose (question, response) the history already holds for any reason.
    """
    from datasets.kt_dataset import _resolve_selectmasks

    df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=max_rows)
    has_repeat_col = "is_repeat" in df.columns
    scored = reattempt = split_attempt = 0
    for _, row in df.iterrows():
        responses = _seq(row, "responses")
        if not responses:
            continue
        questions = _seq(row, "questions") or _seq(row, "concepts")
        masks, _ = _resolve_selectmasks(row, responses)
        repeats = _seq(row, "is_repeat") if has_repeat_col else []
        seen = {}
        for i, response in enumerate(responses):
            if response == -1 or i >= len(questions):
                continue
            if masks[i] == 1:
                scored += 1
                if response in seen.get(questions[i], ()):
                    reattempt += 1
                if repeats and i < len(repeats) and repeats[i] == 1:
                    split_attempt += 1
            seen.setdefault(questions[i], set()).add(response)
    return {
        "path": Path(path).name,
        "scored": scored,
        "reattempt": reattempt,
        "split_attempt": split_attempt,
    }


def _files_for(dataset):
    """Every sequence file a run may be scored on, concept- and question-level.

    The windowed files matter as much as the plain ones: they are the protocol
    pykt reports, and they carry the same `is_repeat` expansion, so a filter
    that only covered the plain files would leave the reported number leaking.
    """
    base = ROOT / "data" / dataset
    pairs = [
        ("one_by_one", base / "test_sequences.csv"),
        ("one_by_one+win", base / "test_window_sequences.csv"),
        ("all_in_one", base / "test_sequences_quelevel.csv"),
        ("all_in_one+win", base / "test_window_sequences_quelevel.csv"),
    ]
    return [(mode, p) for mode, p in pairs if p.exists()]


class TestNoTargetLeakage(unittest.TestCase):
    def test_split_attempts_are_never_scored(self):
        from datasets.kt_dataset import SCORE_REPEATED_KC

        if SCORE_REPEATED_KC:
            self.skipTest("KT_SCORE_REPEATED_KC=1 deliberately restores the old behaviour")
        checked = 0
        for dataset in DEFAULT_DATASETS:
            for mode, path in _files_for(dataset):
                with self.subTest(dataset=dataset, mode=mode):
                    result = _apply_repository_mask(path)
                    self.assertGreater(result["scored"], 0, "no scored positions found")
                    self.assertEqual(
                        result["split_attempt"], 0,
                        f"{dataset}/{mode}: {result['split_attempt']}/{result['scored']} "
                        "scored positions are the same attempt split across its KCs",
                    )
                    checked += 1
        self.assertGreater(checked, 0, "no sequence files found to audit")

    def test_both_modes_agree_on_reattempt_rate(self):
        """A gap here means one mode is still counting split attempts as
        genuine re-attempts, i.e. the repeat filter is not doing its job."""
        from datasets.kt_dataset import SCORE_REPEATED_KC

        if SCORE_REPEATED_KC:
            self.skipTest("KT_SCORE_REPEATED_KC=1 deliberately restores the old behaviour")
        for dataset in DEFAULT_DATASETS:
            # Compare only within the same windowing: a windowed file scores
            # one position per window, so its re-attempt rate legitimately
            # differs from the plain file's.  The invariant is between the two
            # concept modes, which must see the same learners either way.
            groups = {"plain": {}, "windowed": {}}
            for mode, path in _files_for(dataset):
                r = _apply_repository_mask(path)
                bucket = "windowed" if mode.endswith("+win") else "plain"
                groups[bucket][mode] = r["reattempt"] / max(r["scored"], 1)
            for bucket, rates in groups.items():
                if len(rates) < 2:
                    continue
                with self.subTest(dataset=dataset, windowing=bucket):
                    spread = max(rates.values()) - min(rates.values())
                    self.assertLess(
                        spread, 0.01,
                        f"{dataset}/{bucket}: re-attempt rate differs across concept "
                        f"modes by {spread:.1%} ({rates})",
                    )


def main(datasets):
    print(f"{'dataset':<20}{'mode':<12}{'scored':>9}{'split':>8}{'reattempt':>11}{'':>4}")
    print("-" * 66)
    failures = 0
    for dataset in datasets:
        files = _files_for(dataset)
        if not files:
            print(f"{dataset:<20}(no sequence files found)")
            continue
        for mode, path in files:
            r = _apply_repository_mask(path)
            pct = 100.0 * r["reattempt"] / max(r["scored"], 1)
            bad = r["split_attempt"] > 0
            failures += bad
            print(
                f"{dataset:<20}{mode:<12}{r['scored']:>9}{r['split_attempt']:>8}"
                f"{pct:>10.1f}%{'  FAIL' if bad else '  ok':>4}"
            )
    print()
    print("split = same attempt expanded across KCs, scored (must be 0)")
    print("reattempt = learner genuinely answered this question the same way before "
          "(informational; the two modes must agree)")
    print()
    print("CLEAN" if failures == 0 else f"LEAKING in {failures} file(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args or DEFAULT_DATASETS))
