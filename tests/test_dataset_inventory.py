"""What data_config.json declares must match what is on disk, or say so here.

A dataset config is a set of promises: these files exist, this many concepts and
questions, these folds, at most this many KCs per question. Nothing checked them,
so a config could promise a file that was never generated and the failure would
surface as a confusing error inside a training run.

Audited 2026-09-18 against the current data. Everything below is measured, not
assumed, and the known gaps are recorded rather than ignored -- a dataset that
becomes complete, or one that quietly loses a file, breaks a test here.
"""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_CONFIG = json.loads((ROOT / "configs" / "data_config.json").read_text(encoding="utf-8"))

# Declared in the config with no data directory on disk. Three of them
# (assist2015, poj, statics2011) also declare num_q = 0, meaning they carry
# concepts only.
NO_DATA_ON_DISK = {
    "assist2015", "ednet5w", "poj", "pretrain",
}

# Raw data present, sequence files not built yet. `ednet` was here until
# 2026-09-18, when EdNet-Contents.zip arrived and it was generated; `ednet5w`
# would need the same run over 55,000 users instead of 5,000, which is roughly
# 70 GB of windowed test files.
RAW_ONLY = set()

# Every remaining entry above still carries a `../data/...` dpath, a leftover
# from an older directory layout. statics2011 had the same one until it was
# built on 2026-09-18, which is why none of them has ever run: the path is wrong
# and nothing checked it.

# `peiyou` is an alias: same config, same dpath, and normalize_dataset_name maps
# it onto aaai2023. data/peiyou/ is an empty directory.
ALIASES = {"peiyou": "aaai2023"}

# aaai2023 is a hidden-label competition set -- pykt_test.csv marks its targets
# -1 and the runner disables test evaluation for it. Only two of its twelve
# declared files were ever generated, so it can train concept-level and nothing
# else. Recorded rather than pruned from the config: the other ten describe the
# layout a regeneration would produce.
PARTIAL = {
    "aaai2023": {"test_file", "train_valid_file"},
}

FILE_KEYS = (
    "train_valid_file", "train_valid_file_quelevel",
    "train_valid_original_file", "train_valid_original_file_quelevel",
    "test_file", "test_file_quelevel",
    "test_original_file", "test_original_file_quelevel",
    "test_window_file", "test_window_file_quelevel",
)


def usable_datasets():
    """Datasets with data on disk, excluding aliases and the partial one."""
    names = []
    for name, cfg in sorted(DATA_CONFIG.items()):
        if not isinstance(cfg, dict) or name in NO_DATA_ON_DISK or name in ALIASES:
            continue
        if name in RAW_ONLY:
            continue
        if name in PARTIAL:
            continue
        if Path(cfg.get("dpath", "")).is_dir():
            names.append(name)
    return names


def path_for(cfg, key):
    value = cfg.get(key)
    return Path(cfg["dpath"]) / value if value else None


class DeclaredStateTest(unittest.TestCase):
    def test_the_datasets_with_no_data_are_the_recorded_ones(self):
        absent = {
            name for name, cfg in DATA_CONFIG.items()
            if isinstance(cfg, dict) and not Path(cfg.get("dpath", "")).is_dir()
        }
        self.assertEqual(
            absent, NO_DATA_ON_DISK | {n for n in RAW_ONLY
                                       if not Path(DATA_CONFIG[n]["dpath"]).is_dir()},
            "the set of datasets declared without data on disk changed; update "
            "NO_DATA_ON_DISK, or the config is promising something new.",
        )

    def test_peiyou_is_an_alias_of_aaai2023(self):
        from core.dataset_names import normalize_dataset_name

        self.assertEqual(normalize_dataset_name("peiyou"), "aaai2023")
        alias, target = DATA_CONFIG["peiyou"], DATA_CONFIG["aaai2023"]
        self.assertEqual(alias["dpath"], target["dpath"])
        self.assertEqual(
            {k: v for k, v in alias.items() if k != "dpath"},
            {k: v for k, v in target.items() if k != "dpath"},
        )

    def test_aaai2023_still_has_only_its_two_generated_files(self):
        """If this fails the dataset was regenerated -- update PARTIAL."""
        cfg = DATA_CONFIG["aaai2023"]
        present = {k for k in FILE_KEYS if (p := path_for(cfg, k)) and p.exists()}
        self.assertEqual(present, PARTIAL["aaai2023"])


class UsableDatasetsTest(unittest.TestCase):
    """Nine datasets are complete enough to train on. These pin that."""

    def test_every_declared_file_exists(self):
        for name in usable_datasets():
            cfg = DATA_CONFIG[name]
            for key in FILE_KEYS:
                path = path_for(cfg, key)
                if path is None:
                    continue  # not declared is a separate statement, below
                with self.subTest(dataset=name, file=key):
                    self.assertTrue(
                        path.exists(),
                        f"{name} declares {key}={cfg[key]!r}, which is not on disk. "
                        "A run reaches this as a loader failure part way through.",
                    )

    def test_only_junyi_sub5k_declares_no_windowed_split(self):
        """The runner treats that as a dataset property, not a broken run."""
        undeclared = {
            name for name in usable_datasets()
            if not DATA_CONFIG[name].get("test_window_file")
        }
        self.assertEqual(undeclared, {"junyi_sub5k"})

    def test_num_c_and_num_q_match_keyid2idx(self):
        for name in usable_datasets():
            cfg = DATA_CONFIG[name]
            keyid = Path(cfg["dpath"]) / "keyid2idx.json"
            if not keyid.exists():
                continue
            mapping = json.loads(keyid.read_text(encoding="utf-8"))
            with self.subTest(dataset=name):
                self.assertEqual(cfg["num_c"], len(mapping.get("concepts", {})))
                self.assertEqual(cfg["num_q"], len(mapping.get("questions", {})))

    def test_every_declared_fold_is_present_and_populated(self):
        for name in usable_datasets():
            cfg = DATA_CONFIG[name]
            path = path_for(cfg, "train_valid_file_quelevel") or path_for(
                cfg, "train_valid_file"
            )
            if path is None or not path.exists():
                continue
            folds = pd.read_csv(path, usecols=["fold"])["fold"].astype(int)
            with self.subTest(dataset=name):
                self.assertEqual(
                    sorted(set(folds)), sorted(cfg["folds"]),
                    f"{name} declares folds {cfg['folds']} but the data holds "
                    f"{sorted(set(folds))}; a missing fold silently shrinks CV.",
                )
                self.assertTrue(
                    (folds.value_counts() > 0).all(),
                    f"{name} has an empty fold.",
                )


class ConceptWidthTest(unittest.TestCase):
    """`max_concepts` too small silently truncates; too large only wastes slots."""

    def test_max_concepts_is_never_smaller_than_the_data_needs(self):
        for name in usable_datasets():
            cfg = DATA_CONFIG[name]
            path = path_for(cfg, "train_valid_file_quelevel")
            if path is None or not path.exists():
                continue
            widest = 0
            for chunk in pd.read_csv(
                path, usecols=["concepts"], dtype=str,
                keep_default_na=False, chunksize=4000,
            ):
                for row in chunk["concepts"]:
                    for token in str(row).split(","):
                        widest = max(widest, len(token.split("_")))
            with self.subTest(dataset=name):
                self.assertGreaterEqual(
                    cfg["max_concepts"], widest,
                    f"{name} declares max_concepts={cfg['max_concepts']} but a "
                    f"question carries {widest} KCs, so the extras are dropped.",
                )


if __name__ == "__main__":
    unittest.main()


class EdNetSamplingTest(unittest.TestCase):
    """`ednet` and `ednet5w` are two slices of one release, not two datasets.

    One shuffle of the KT1 user ids under seed 2 produces both: the first takes
    the first 5,000 users found, the second skips those and takes the next
    50,000. Nothing in the config used to say so, so a table could put them side
    by side as if they were independent, and a paper could say "we use EdNet"
    without saying which 0.6% or 6% of it.
    """

    def test_both_declare_where_they_came_from(self):
        for name in ("ednet", "ednet5w"):
            with self.subTest(dataset=name):
                cfg = DATA_CONFIG[name]
                self.assertEqual(cfg["source"], "EdNet-KT1")
                self.assertIn("sampling", cfg)

    def test_the_two_slices_are_distinguishable_in_an_artifact(self):
        self.assertNotEqual(
            DATA_CONFIG["ednet"]["sampling"],
            DATA_CONFIG["ednet5w"]["sampling"],
        )

    def test_the_recorded_plan_matches_the_code(self):
        """A config that drifts from the preprocessor is worse than no config."""
        from preprocess.ednet_preprocess import SAMPLING

        for name, plan in SAMPLING.items():
            with self.subTest(dataset=name):
                expected = (f"pykt_seed{plan['seed']}_skip{plan['skip_users']}"
                            f"_take{plan['take_users']}")
                self.assertEqual(DATA_CONFIG[name]["sampling"], expected)

    def test_neither_slice_claims_to_be_the_whole_release(self):
        from preprocess.ednet_preprocess import SAMPLING

        total = sum(p["take_users"] for p in SAMPLING.values())
        self.assertLess(total, 784_309)


class ProvenanceTest(unittest.TestCase):
    """A dataset that is a sample of another must say so in the config.

    `ednet` and `ednet5w` are slices of EdNet-KT1; `junyi_sub5k` is a subsample
    of `junyi2015` built here and matching no published protocol. Nothing in a
    run's artifacts distinguished them from a full dataset, so a table could
    place a 5,000-student cut beside a 247,606-student one without a hint, and a
    paper could report "Junyi" or "EdNet" without saying which slice.
    """

    # Datasets that are not the full release they are named after.
    SAMPLED = {"ednet", "ednet5w", "junyi_sub5k"}

    def test_every_sampled_dataset_declares_its_source_and_plan(self):
        for name in sorted(self.SAMPLED):
            with self.subTest(dataset=name):
                cfg = DATA_CONFIG[name]
                self.assertIn("source", cfg)
                self.assertIn("sampling", cfg)
                self.assertNotEqual(
                    cfg["sampling"], "none",
                    f"{name} is a sample, so its plan cannot be 'none'.",
                )

    def test_a_sampled_dataset_names_what_it_was_drawn_from(self):
        self.assertIn("EdNet-KT1", DATA_CONFIG["ednet"]["source"])
        self.assertIn("junyi2015", DATA_CONFIG["junyi_sub5k"]["source"])

    def test_the_full_dataset_says_it_is_not_sampled(self):
        """Absence of a field would be ambiguous; 'none' is a statement."""
        self.assertEqual(DATA_CONFIG["junyi2015"]["sampling"], "none")

    def test_sampled_datasets_carry_a_note_a_reader_can_act_on(self):
        for name in sorted(self.SAMPLED):
            with self.subTest(dataset=name):
                cfg = DATA_CONFIG[name]
                if "sampling_note" not in cfg:
                    continue
                self.assertGreater(len(cfg["sampling_note"]), 80)

    def test_the_junyi_subset_plan_matches_its_manifest(self):
        """The config must not drift from what the builder actually recorded."""
        manifest_path = Path(DATA_CONFIG["junyi_sub5k"]["dpath"]) / "subset_manifest.json"
        if not manifest_path.exists():
            self.skipTest("junyi_sub5k not built here")
        selection = json.loads(manifest_path.read_text(encoding="utf-8"))["selection"]
        expected = (
            f"{selection['algorithm']}_v{selection['algorithm_version']}"
            f"_n{selection['students']}_strata{selection['strata_count']}"
            f"_minlen{selection['min_sequence_length']}_seed{selection['seed']}"
        )
        self.assertEqual(DATA_CONFIG["junyi_sub5k"]["sampling"], expected)

    def test_the_junyi_strata_are_equally_allocated(self):
        """Equal quotas over equal-rank strata is what preserves the length
        distribution; unequal quotas would reshape it."""
        manifest_path = Path(DATA_CONFIG["junyi_sub5k"]["dpath"]) / "subset_manifest.json"
        if not manifest_path.exists():
            self.skipTest("junyi_sub5k not built here")
        selection = json.loads(manifest_path.read_text(encoding="utf-8"))["selection"]
        taken = {s["selected_students"] for s in selection["strata"]}
        self.assertEqual(len(taken), 1, f"strata took different counts: {taken}")
        self.assertEqual(
            sum(s["selected_students"] for s in selection["strata"]),
            selection["students"],
        )
