"""The ASSISTments `keep_scaffolding` switch must work in both directions.

`keep_scaffolding=True` is pyKT's preprocessing and the project default: the
raw rows, unfiltered, which is what every published number on these datasets is
computed from.

`keep_scaffolding=False` drops the leak. Scaffolding sub-problems are generated
only after a wrong answer on the main problem, so leaving them in carries the
previous label; assist2017 is additionally an action log in which a failed
problem is retried until solved, so it must also be collapsed to one row per
problem attempt. Correct data, but not comparable to the literature.

These tests pass `keep_scaffolding` explicitly and so do not depend on which
value is the default.
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess.assist2009_preprocess import read_data_from_csv as read_2009
from preprocess.assist2012_preprocess import read_data_from_csv as read_2012
from preprocess.assist2017_preprocess import read_data_from_csv as read_2017


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sequence_lengths(path):
    """Parse write_txt output: 6 lines per user, header is `uid,seq_len`."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    return [int(lines[i * 6].split(",")[1]) for i in range(len(lines) // 6)]


class ScaffoldingFilterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, reader, raw_path, keep):
        out = self.tmp / f"out_{int(keep)}.txt"
        reader(str(raw_path), str(out), keep_scaffolding=keep)
        return sequence_lengths(out)

    def test_assist2009_drops_scaffolding_by_default(self):
        raw = self.tmp / "a09.csv"
        fields = ["order_id", "user_id", "problem_id", "original", "correct", "skill_id"]
        rows = []
        for i, (orig, correct) in enumerate(
            [("1", "1"), ("1", "0"), ("0", "0"), ("0", "1"), ("1", "1")]
        ):
            rows.append({"order_id": str(i), "user_id": "u1", "problem_id": f"p{i}",
                         "original": orig, "correct": correct, "skill_id": "s1"})
        write_csv(raw, fields, rows)

        self.assertEqual(self._run(read_2009, raw, False), [3])
        self.assertEqual(self._run(read_2009, raw, True), [5])

    def test_assist2012_drops_scaffolding_by_default(self):
        raw = self.tmp / "a12.csv"
        fields = ["user_id", "skill_id", "start_time", "problem_id", "correct",
                  "ms_first_response", "original"]
        rows = []
        for i, orig in enumerate([1, 1, 0, 0, 1]):
            rows.append({"user_id": "u1", "skill_id": "s1",
                         "start_time": f"2012-09-01 10:0{i}:00",
                         "problem_id": f"p{i}", "correct": i % 2,
                         "ms_first_response": 1000 + i, "original": orig})
        write_csv(raw, fields, rows)

        self.assertEqual(self._run(read_2012, raw, False), [3])
        self.assertEqual(self._run(read_2012, raw, True), [5])

    def test_assist2017_drops_scaffolding_and_collapses_retries(self):
        raw = self.tmp / "a17.csv"
        fields = ["studentId", "problemId", "assignmentId", "skill", "correct",
                  "startTime", "timeTaken", "original", "scaffold"]
        rows = [
            # p1 retried three times until solved -> collapses to the first action
            {"studentId": "s1", "problemId": "p1", "assignmentId": "a1", "skill": "k1",
             "correct": 0, "startTime": 100, "timeTaken": 5, "original": 1, "scaffold": 0},
            {"studentId": "s1", "problemId": "p1", "assignmentId": "a1", "skill": "k1",
             "correct": 0, "startTime": 110, "timeTaken": 5, "original": 1, "scaffold": 0},
            {"studentId": "s1", "problemId": "p1", "assignmentId": "a1", "skill": "k1",
             "correct": 1, "startTime": 120, "timeTaken": 5, "original": 1, "scaffold": 0},
            # two scaffolding rows spawned by the failure
            {"studentId": "s1", "problemId": "p1s", "assignmentId": "a1", "skill": "k1",
             "correct": 1, "startTime": 130, "timeTaken": 5, "original": 0, "scaffold": 1},
            {"studentId": "s1", "problemId": "p1s2", "assignmentId": "a1", "skill": "k1",
             "correct": 1, "startTime": 140, "timeTaken": 5, "original": 0, "scaffold": 1},
            # a clean single-action problem
            {"studentId": "s1", "problemId": "p2", "assignmentId": "a1", "skill": "k2",
             "correct": 1, "startTime": 150, "timeTaken": 5, "original": 1, "scaffold": 0},
        ]
        write_csv(raw, fields, rows)

        self.assertEqual(self._run(read_2017, raw, False), [2])
        self.assertEqual(self._run(read_2017, raw, True), [6])

    def test_assist2017_keeps_first_attempt_label(self):
        """The collapsed row must carry first-attempt correctness, not the retry."""
        raw = self.tmp / "a17b.csv"
        fields = ["studentId", "problemId", "assignmentId", "skill", "correct",
                  "startTime", "timeTaken", "original", "scaffold"]
        rows = [
            {"studentId": "s1", "problemId": "p1", "assignmentId": "a1", "skill": "k1",
             "correct": 0, "startTime": 200, "timeTaken": 5, "original": 1, "scaffold": 0},
            {"studentId": "s1", "problemId": "p1", "assignmentId": "a1", "skill": "k1",
             "correct": 1, "startTime": 210, "timeTaken": 5, "original": 1, "scaffold": 0},
        ]
        write_csv(raw, fields, rows)

        out = self.tmp / "out.txt"
        read_2017(str(raw), str(out), keep_scaffolding=False)
        with open(out, encoding="utf-8") as f:
            lines = f.read().rstrip("\n").split("\n")
        self.assertEqual(lines[0].split(",")[1], "1")   # one interaction survives
        self.assertEqual(lines[3], "0")                 # and its label is the first attempt

    def test_same_assignment_different_problems_are_not_collapsed(self):
        raw = self.tmp / "a17c.csv"
        fields = ["studentId", "problemId", "assignmentId", "skill", "correct",
                  "startTime", "timeTaken", "original", "scaffold"]
        rows = [
            {"studentId": "s1", "problemId": f"p{i}", "assignmentId": "a1", "skill": "k1",
             "correct": 1, "startTime": 300 + i, "timeTaken": 5, "original": 1, "scaffold": 0}
            for i in range(4)
        ]
        write_csv(raw, fields, rows)
        self.assertEqual(self._run(read_2017, raw, False), [4])


if __name__ == "__main__":
    unittest.main()
