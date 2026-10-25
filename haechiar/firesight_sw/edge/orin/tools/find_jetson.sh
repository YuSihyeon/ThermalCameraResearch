#!/usr/bin/env bash
# Find a Jetson on the local network from the Mac.
# NVIDIA boards answer from the 00:04:4b and 48:b0:2d MAC ranges, which is a
# more reliable signal than the hostname.
set -uo pipefail

subnet="${SUBNET:-}"
if [[ -z "$subnet" ]]; then
  local_ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en10 2>/dev/null)
  [[ -z "$local_ip" ]] && { echo "no active network interface"; exit 1; }
  subnet="${local_ip%.*}"
fi

echo "sweeping ${subnet}.1-254 ..."
for host in $(seq 1 254); do
  (ping -c1 -W120 "${subnet}.${host}" >/dev/null 2>&1 &)
done
python3 -c "import time; time.sleep(6)"

echo
echo "NVIDIA-range devices:"
found=$(arp -a | grep -iE "([[:space:]]|:)0:4:4b:|00:04:4b:|48:b0:2d:")
if [[ -n "$found" ]]; then
  echo "$found" | sed 's/^/  /'
  echo
  echo "Connect with:  ssh <account>@<the address above>"
else
  echo "  none found"
  echo
  echo "Check that the Jetson is powered on and its Ethernet cable reaches the"
  echo "same router as this Mac. If it is on a direct cable to the USB adapter,"
  echo "run with SUBNET=169.254 or set static addresses on both ends."
fi

echo
echo "all devices seen:"
arp -a | grep -v incomplete | sed 's/^/  /'
