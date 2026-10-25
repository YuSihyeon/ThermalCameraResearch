#!/usr/bin/env bash
# Run this on the Jetson Orin Nano. It checks each prerequisite in order and
# stops at the first thing that would make the pipeline fail, so a bad result
# points at one cause instead of a pile of them.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PYTHON:-python3}"
CAMERA="${CAMERA:-/dev/video0}"
FRAMES="${FRAMES:-300}"
OUT="${OUT:-metrics_orin.json}"
fail=0

section() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()      { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()     { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
warn()    { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

section "1. Board"
if [[ -f /etc/nv_tegra_release ]]; then
  ok "$(head -1 /etc/nv_tegra_release)"
else
  warn "not a Jetson (no /etc/nv_tegra_release) - continuing anyway"
fi
command -v nvpmodel >/dev/null && ok "power mode: $(sudo nvpmodel -q 2>/dev/null | tr '\n' ' ')" \
  || warn "nvpmodel not found"

section "2. Python and packages"
ok "$($PY -V 2>&1)"
for module in numpy cv2 torch; do
  if version=$($PY -c "import $module; print(getattr($module,'__version__','?'))" 2>/dev/null); then
    ok "$module $version"
  else
    bad "$module is missing"
  fi
done

section "3. CUDA"
cuda_report=$($PY - <<'PY' 2>/dev/null
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
)
if [[ "$(echo "$cuda_report" | head -1)" == "True" ]]; then
  ok "CUDA available on $(echo "$cuda_report" | tail -1)"
else
  bad "torch.cuda.is_available() is False - install the JetPack-matched NVIDIA wheel, not PyPI torch"
fi

section "4. Checkpoint"
# The checkpoint is shared with the teed package; fall back to a local copy.
CKPT="../teed/models/5_model.pth"
[[ -f "$CKPT" ]] || CKPT="models/teed/5_model.pth"
EXPECT="0322caf70f588355aaaf59c2bf5872b21a4b7e9f679971a7a3bb1f69b56a01ba"
if [[ -f "$CKPT" ]]; then
  actual=$(sha256sum "$CKPT" | cut -d' ' -f1)
  [[ "$actual" == "$EXPECT" ]] && ok "5_model.pth sha256 matches" \
    || bad "5_model.pth sha256 mismatch: $actual"
else
  bad "$CKPT not found - copy it from the firesight_sw repo (edge/teed/models/5_model.pth)"
fi

section "5. Camera"
if command -v v4l2-ctl >/dev/null; then
  v4l2-ctl --list-devices 2>/dev/null | sed 's/^/  /'
else
  warn "v4l2-ctl not installed (sudo apt-get install v4l-utils)"
fi
if [[ -e "$CAMERA" ]]; then
  ok "$CAMERA exists"
  groups | grep -qw video && ok "user is in the video group" \
    || warn "user is not in the video group: sudo usermod -aG video \$USER, then log out and back in"
else
  bad "$CAMERA not found - set CAMERA=/dev/videoN"
fi

section "6. Displays"
if command -v xrandr >/dev/null; then
  $PY run_pipeline.py --list-displays 2>/dev/null | sed 's/^/  /'
else
  warn "xrandr not available (headless session) - --screen will not work, use --window-x"
fi

if [[ $fail -ne 0 ]]; then
  printf '\n\033[31mStopping: fix the FAIL items above before measuring.\033[0m\n'
  exit 1
fi

section "7. Measured run ($FRAMES frames, CUDA FP16)"
sudo jetson_clocks 2>/dev/null && ok "jetson_clocks applied" || warn "could not apply jetson_clocks"
$PY run_pipeline.py \
  --source "$CAMERA" \
  --device cuda --half \
  --width 320 --height 240 \
  --headless --max-frames "$FRAMES" \
  --metrics "$OUT" >/dev/null 2>&1

if [[ -f "$OUT" ]]; then
  ok "wrote $OUT"
  $PY - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"\n  frames {d['processed_frames']}   effective fps {d['effective_fps']:.1f}")
print(f"  {'stage':<14}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}")
for key, value in d["latency_ms"].items():
    print(f"  {key:<14}{value['mean']:>8.2f}{value['p50']:>9.2f}"
          f"{value['p95']:>9.2f}{value['max']:>9.2f}")
print("\n  Send this file back to fill in RUNBOOK.md section 4.3.")
PY
else
  bad "the measured run produced no metrics; rerun without --headless to see the error"
fi
