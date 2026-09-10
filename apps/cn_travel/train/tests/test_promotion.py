"""Checks for promoting selected training artifacts into the runtime bundle."""

import importlib.util

from project_paths import TRAIN_ROOT


def test_selected_training_artifacts_match_the_runtime_bundle():
    script = TRAIN_ROOT / "promote_runtime.py"
    spec = importlib.util.spec_from_file_location("promote_runtime_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_promoted()
