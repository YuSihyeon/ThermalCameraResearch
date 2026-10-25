from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("arguments, expected", [
    (["wizard"], "true"), (["review", "--calibration", "result.yaml"], "true"),
    (["wizard", "--source", "usb"], "false"), (["review", "--source=usb"], "false"),
    (["self-check"], "false"), (["capture", "--source", "ros"], "true"),
    (["wizard", "--source", "ros", "--source=usb"], "false"),
])
def test_linux_launcher_selects_same_source_as_cli(arguments, expected):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    launcher = Path(__file__).parents[2] / "run.sh"
    script = launcher.read_text(encoding="utf-8")
    # Execute the launcher's real argument-selection block without bootstrapping
    # dependencies or opening a device. Bootstrap is exercised by Humble CI.
    selection = script[script.index("calib_ros=false"):script.index('if "$calib_ros"; then')]
    result = subprocess.run([bash, "-c", selection + '\nprintf "%s" "$calib_ros"', "run.sh", *arguments],
                            text=True, capture_output=True, check=True)
    assert result.stdout == expected
