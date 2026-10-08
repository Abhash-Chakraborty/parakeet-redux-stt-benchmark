"""Kaggle kernel, run 2: re-run only the reference configurations that failed in run 1 on tooling bugs
(faster-whisper/PyAV clash; NeMo Parakeet Unified missing validation_ds). Same data, same T4, same worker
format; fixed worker in the stt-benchmark-code dataset.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATA = str(Path(glob.glob("/kaggle/input/**/stt-benchmark-audio/**/units.csv", recursive=True)[0]).parent)
BENCH = glob.glob("/kaggle/input/**/stt-benchmark-code/**/bench.py", recursive=True)[0]
OUT = Path("/kaggle/working/out")
VENVS = Path("/kaggle/working/venvs")
(OUT / "logs").mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "main.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=3600, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def run(py, backend, system, sset, extra_env=None):
    tag = f"{system}__gpu_{sset}"
    out = OUT / "raw" / system / f"gpu_{sset}{'_t4' if sset == 'speed' else ''}.jsonl"
    t = time.time()
    try:
        rc, o = sh(f"{py} -u {BENCH} --backend {backend} --system {system} --device gpu --set {sset} --threads 4 "
                   f"--data {DATA} --out {out}", 5400, dict(os.environ, **(extra_env or {})))
    except subprocess.TimeoutExpired:
        rc, o = -9, "timeout"
    (OUT / "logs" / f"{tag}.log").write_text(o[-30000:])
    ok = rc == 0
    with open(OUT / "status.jsonl", "a") as f:
        f.write(json.dumps({"backend": backend, "system": system, "device": "gpu", "set": sset, "ok": ok, "rc": rc,
                            "seconds": round(time.time() - t),
                            "error": None if ok else (o.strip().splitlines() or [""])[-1][:400]}) + "\n")
    log(f"{'OK  ' if ok else 'FAIL'} {backend:6s} {tag} ({time.time() - t:.0f}s)")


log("gpu: " + sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", 60)[1].strip())
sh(f"{sys.executable} -m pip install -q uv", 600)
path = VENVS / "fw"
rc, o = sh(f"uv venv -q --python 3.12 {path} && uv pip install -q --python {path}/bin/python faster-whisper "
           f"'nvidia-cublas-cu12' 'nvidia-cudnn-cu12==9.*' numpy soundfile psutil nvidia-ml-py", 3600)
log(f"venv fw rc={rc}")
libs = glob.glob(str(path / "lib/python3.12/site-packages/nvidia/*/lib"))
fw_env = {"LD_LIBRARY_PATH": ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])}
for system in ("wturbo_fw", "wlv3_fw"):
    for sset in ("accuracy", "speed"):
        run(f"{path}/bin/python", "fw", system, sset, fw_env)

sh(f"{sys.executable} -m pip install -q nvidia-ml-py psutil soundfile 'nemo_toolkit[asr]>=2.7'", 3600)
for sset in ("accuracy", "speed"):
    run(sys.executable, "nemo", "unified_nemo", sset)
log("ALL DONE")
