#!/usr/bin/env bash
# Accuracy pass (PROTOCOL 6.6): every transcribe.cpp system on GPU, one fresh process each, randomized order.
set -e
cd "$(dirname "$0")/.."
SUBSETS=${1:-A,B,B2,C}
for s in $(printf "tdt_f16\ntdt_q8\nunified_q8\nwlv3_q8\nwturbo_q8\nmoon_f16\n" | envs/tcpp/Scripts/python.exe -c "import sys,random; l=sys.stdin.read().split(); random.Random(42).shuffle(l); print(' '.join(l))"); do
  echo "=== $s gpu $(date +%T)"
  envs/tcpp/Scripts/python.exe scripts/run_local.py --system $s --device gpu --set accuracy --subsets $SUBSETS | tail -1 | cut -c1-200
done
echo "ALL DONE $(date +%T)"
