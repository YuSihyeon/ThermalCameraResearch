#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH_ROOT="$(cd "$HERE/.." && pwd)"
TARGET="$DEPTH_ROOT/third_party/RIDERS"

mkdir -p "$DEPTH_ROOT/third_party"
if [[ ! -d "$TARGET/.git" ]]; then
  git clone https://github.com/MMOCKING/RIDERS.git "$TARGET"
else
  echo "already cloned: $TARGET"
fi

echo
printf '%s\n' \
  "RIDERS source ready: $TARGET" \
  "For full RIDERS, sensor projection plus RC-Net/SML checkpoints are still required." \
  "The FireSight folder can already run prepare_inputs.py and run_stage1_alignment.py."
