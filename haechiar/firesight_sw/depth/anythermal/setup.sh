#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH_ROOT="$(cd "$HERE/.." && pwd)"
TARGET="$DEPTH_ROOT/third_party/AnyThermal"

mkdir -p "$DEPTH_ROOT/third_party"
if [[ ! -d "$TARGET/.git" ]]; then
  git clone https://github.com/castacks/AnyThermal.git "$TARGET"
else
  echo "already cloned: $TARGET"
fi

git -C "$TARGET" submodule update --init baselines/depth/BridgeMultiSpectralDepth

echo
printf '%s\n' \
  "AnyThermal source ready: $TARGET" \
  "Download anythermal_checkpoints.zip using the official README and place its contents under:" \
  "$TARGET/pretrained_checkpoints/" \
  "Depth also requires pytorch3d built for your PyTorch/CUDA environment."
