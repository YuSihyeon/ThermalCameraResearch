#!/usr/bin/env bash
set -euo pipefail
calib_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
    set -- wizard --source ros
fi
calib_ros=false
calib_source=''
calib_previous=''
for calib_arg in "$@"; do
    if [[ "$calib_previous" == --source ]]; then
        calib_source="$calib_arg"
    elif [[ "$calib_arg" == --source=* ]]; then
        calib_source="${calib_arg#--source=}"
    fi
    calib_previous="$calib_arg"
done
if [[ "$calib_source" == ros || ( -z "$calib_source" && ( "$1" == wizard || "$1" == review ) ) ]]; then
    calib_ros=true
fi
if "$calib_ros"; then
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        set +u
        source /opt/ros/humble/setup.bash
        set -u
    fi
    calib_python=/usr/bin/python3
    if ! "$calib_python" -c 'import cv2, numpy, yaml, rclpy; from sensor_msgs.msg import Image' 2>/dev/null; then
        echo 'ROS prerequisites missing. Install ROS 2 Humble, then:' >&2
        echo 'sudo apt install python3-opencv python3-numpy python3-yaml ros-humble-rclpy ros-humble-sensor-msgs' >&2
        exit 2
    fi
    export PYTHONPATH="$calib_root/calib${PYTHONPATH:+:$PYTHONPATH}"
else
    calib_python="$calib_root/.venv/bin/python"
    if [[ ! -x "$calib_python" ]]; then
        python3 -m venv "$calib_root/.venv" || { echo 'Install Python 3.10+ and python3-venv first.' >&2; exit 2; }
    fi
    calib_hash="$(sha256sum "$calib_root/pyproject.toml" | cut -d ' ' -f 1)"
    calib_marker="$calib_root/.venv/firesight-bootstrap.txt"
    if [[ ! -f "$calib_marker" || "$(cat "$calib_marker")" != "$calib_hash" ]]; then
        "$calib_python" -m pip install -e "$calib_root"
        printf '%s\n' "$calib_hash" > "$calib_marker"
    fi
fi
exec "$calib_python" -m firesight_calib.cli "$@"
