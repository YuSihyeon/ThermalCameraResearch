"""Check evaluator startup with PATH and explicit Git executables."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import evaluate


class BaselineLoaded(Exception):
    pass


def main() -> None:
    executable = shutil.which("git")
    assert executable is not None, "Git must be installed to evaluate a baseline"
    load_baseline = evaluate.load_baseline_hud

    def inspect_baseline(git: Path, output: Path):
        module = load_baseline(git, output)
        assert callable(module.HUDRenderer)
        expected = evaluate.run_git(
            git, ["show", f"{evaluate.BASELINE_REVISION}:edge/orin/hud.py"]
        )
        assert (output / "_baseline_hud_HEAD.py").read_text(encoding="utf-8") == expected
        raise BaselineLoaded

    with tempfile.TemporaryDirectory() as directory:
        for flags in ([], ["--git", executable]):
            argv = ["evaluate.py", "--output-dir", directory, *flags]
            with patch.object(sys, "argv", argv), patch.object(
                evaluate, "load_baseline_hud", side_effect=inspect_baseline
            ):
                try:
                    evaluate.main()
                except BaselineLoaded:
                    pass
                else:
                    raise AssertionError("Expected baseline startup boundary")

        missing = str(Path(directory) / "missing-git-executable")
        argv = ["evaluate.py", "--output-dir", directory, "--git", missing]
        with patch.object(sys, "argv", argv), patch.object(
            evaluate, "load_baseline_hud"
        ) as baseline:
            try:
                evaluate.main()
            except RuntimeError as error:
                assert "git executable not found" in str(error)
            else:
                raise AssertionError("Missing Git executable was accepted")
            baseline.assert_not_called()
    print("PASS: PATH Git, explicit executable, pinned baseline, missing executable")


if __name__ == "__main__":
    main()
