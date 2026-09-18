"""Caller-supplied directories from the WebUI must stay inside the project.

`save_dir` and `cv_run_dir` arrive in an HTTP body and become filesystem paths:
the first is created and written to, the second is walked by `collect_metrics`
looking for metrics files. Unchecked, an absolute path was used as given and a
relative one could climb out with `..`, so a request was a write primitive
anywhere the process has permission and a read primitive over anything it can
see.

Reachability is limited today -- the server binds to 127.0.0.1 and registers no
CORS middleware -- but `serve_web.py --host` exists, and containment is cheaper
to add than to discover the need for.

The escape hatch is KT_WEBUI_ALLOWED_ROOTS, because writing checkpoints to a
larger disk is a real thing to want and blocking it outright would push people
back to editing the code.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webui.runner import (
    DEFAULT_SAVE_BASE,
    ROOT as RUNNER_ROOT,
    resolve_within_allowed_roots,
)


class InsideTheProjectTest(unittest.TestCase):
    def test_no_value_falls_back_to_the_default(self):
        resolved = resolve_within_allowed_roots(None, DEFAULT_SAVE_BASE, "save_dir")
        self.assertEqual(resolved, Path(DEFAULT_SAVE_BASE).resolve())

    def test_a_relative_path_lands_under_the_project(self):
        resolved = resolve_within_allowed_roots("saved_model/mine", DEFAULT_SAVE_BASE, "save_dir")
        self.assertEqual(resolved, (RUNNER_ROOT / "saved_model" / "mine").resolve())

    def test_an_absolute_path_inside_the_project_is_allowed(self):
        target = RUNNER_ROOT / "saved_model" / "explicit"
        self.assertEqual(
            resolve_within_allowed_roots(str(target), DEFAULT_SAVE_BASE, "save_dir"),
            target.resolve(),
        )

    def test_the_project_root_itself_is_allowed(self):
        self.assertEqual(
            resolve_within_allowed_roots(str(RUNNER_ROOT), DEFAULT_SAVE_BASE, "save_dir"),
            RUNNER_ROOT.resolve(),
        )


class OutsideTheProjectTest(unittest.TestCase):
    def _rejects(self, value):
        with self.assertRaises(ValueError) as err:
            resolve_within_allowed_roots(value, DEFAULT_SAVE_BASE, "save_dir")
        return str(err.exception)

    def test_a_relative_path_cannot_climb_out(self):
        message = self._rejects("../../../tmp/evil")
        self.assertIn("outside the project", message)

    def test_an_absolute_path_outside_is_refused(self):
        self._rejects(str(Path(tempfile.gettempdir()) / "kt-escape"))

    def test_a_path_that_only_looks_contained_is_refused(self):
        """A sibling directory whose name starts with the project's."""
        self._rejects(str(RUNNER_ROOT.parent / (RUNNER_ROOT.name + "-elsewhere")))

    def test_the_message_names_the_field_and_the_way_out(self):
        message = self._rejects("../../../tmp/evil")
        self.assertIn("save_dir", message)
        self.assertIn("KT_WEBUI_ALLOWED_ROOTS", message)

    def test_the_label_follows_the_field_being_checked(self):
        with self.assertRaises(ValueError) as err:
            resolve_within_allowed_roots("../../../tmp/evil", None, "cv_run_dir")
        self.assertIn("cv_run_dir", str(err.exception))


class AllowedRootsEscapeHatchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.outside = Path(self._tmp.name).resolve()
        self._saved = os.environ.get("KT_WEBUI_ALLOWED_ROOTS")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("KT_WEBUI_ALLOWED_ROOTS", None)
        else:
            os.environ["KT_WEBUI_ALLOWED_ROOTS"] = self._saved
        self._tmp.cleanup()

    def test_an_allowed_root_permits_writing_outside(self):
        os.environ["KT_WEBUI_ALLOWED_ROOTS"] = str(self.outside)
        target = self.outside / "runs"
        self.assertEqual(
            resolve_within_allowed_roots(str(target), DEFAULT_SAVE_BASE, "save_dir"),
            target.resolve(),
        )

    def test_an_allowed_root_does_not_open_everything_else(self):
        os.environ["KT_WEBUI_ALLOWED_ROOTS"] = str(self.outside)
        with self.assertRaises(ValueError):
            resolve_within_allowed_roots(
                str(self.outside.parent / "somewhere-else"), DEFAULT_SAVE_BASE, "save_dir"
            )

    def test_several_roots_can_be_listed(self):
        os.environ["KT_WEBUI_ALLOWED_ROOTS"] = os.pathsep.join(
            [str(self.outside / "a"), str(self.outside)]
        )
        self.assertEqual(
            resolve_within_allowed_roots(str(self.outside / "b"), DEFAULT_SAVE_BASE, "save_dir"),
            (self.outside / "b").resolve(),
        )


class ApiSurfaceTest(unittest.TestCase):
    def test_create_job_reports_a_rejection_as_a_client_error(self):
        """runner raises ValueError; app.py already maps that to HTTP 400."""
        source = (ROOT / "webui" / "app.py").read_text(encoding="utf-8")
        self.assertIn("except ValueError as exc:", source)
        self.assertIn("status_code=400", source)


if __name__ == "__main__":
    unittest.main()
