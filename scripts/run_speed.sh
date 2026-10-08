#!/usr/bin/env bash
# Speed/resource pass (PROTOCOL 5-6): every configuration in a fresh process, randomized order (seed 42),
# 60 s cool-down between configurations. Requires a quiet machine (Handy, browsers, heavy apps closed).
cd "$(dirname "$0")/.."
PY=envs/tcpp/Scripts/python.exe
CONFIGS="tdt_f16:cpu:6 tdt_f16:gpu:6 tdt_q8:cpu:6 tdt_q8:gpu:6 unified_q8:cpu:6 unified_q8:gpu:6
wlv3_q8:cpu:6 wlv3_q8:gpu:6 wturbo_q8:cpu:6 wturbo_q8:gpu:6 moon_f16:cpu:6 moon_f16:gpu:6
redux:cpu:4 redux:cpu:6 redux:cpu:12 tdt_q8:cpu:6:docker"
ORDER=$(echo $CONFIGS | tr ' ' '\n' | $PY -c "import sys,random; l=sys.stdin.read().split(); random.Random(42).shuffle(l); print(' '.join(l))")
echo "order: $ORDER"
for c in $ORDER; do
  IFS=: read -r sys dev thr dock <<< "$c"
  echo "=== $c $(date +%T)"
  $PY scripts/run_local.py --system $sys --device $dev --set speed --threads $thr ${dock:+--docker} | tail -1 | cut -c1-160
  sleep 60
done
echo "ALL DONE $(date +%T)"
