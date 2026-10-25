#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH_ROOT="$(cd "$HERE/.." && pwd)"
TARGET="$DEPTH_ROOT/third_party/monother_depth"

mkdir -p "$DEPTH_ROOT/third_party"
if [[ ! -d "$TARGET/.git" ]]; then
  git clone https://github.com/ZuoJiaxing/monother_depth.git "$TARGET"
else
  echo "already cloned: $TARGET"
fi

mkdir -p "$TARGET/checkpoints"
echo
printf '%s\n' \
  "MonoTher-Depth source ready: $TARGET" \
  "Next: prepare the released MonoTher MS2 checkpoint and depth_anything_vitl14.pth." \
  "See $HERE/README.md for exact filenames and run command."
