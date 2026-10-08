"""Kaggle kernel: rerun the speech-to-text benchmark on Kaggle hardware (GPU: NVIDIA T4).

Each runtime gets its own environment; each (system, device, set) runs in a fresh process with a timeout, so
one failure cannot stop the rest. Everything lands in /kaggle/working/out (downloaded afterwards and scored
locally with the same scorer as the laptop run).
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATA = str(Path(glob.glob("/kaggle/input/**/units.csv", recursive=True)[0]).parent)
OUT = Path("/kaggle/working/out")
VENVS = Path("/kaggle/working/venvs")
BENCH = str(Path(DATA) / "bench.py")
NPROC = os.cpu_count()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "main.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=3600, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def status(**kw):
    with open(OUT / "status.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


# ---- 1. hardware and environment -----------------------------------------------------------------------------
env = {}
for name, cmd in [("lscpu", "lscpu"), ("nvidia_smi", "nvidia-smi"), ("free", "free -m"), ("os", "cat /etc/os-release"),
                  ("cpu_flags", "grep -m1 -o -w -E 'avx2|avx512f|avx512_vnni|avx_vnni|amx_tile' /proc/cpuinfo | sort -u")]:
    env[name] = sh(cmd, 60)[1]
env["nproc"] = NPROC
log(f"nproc={NPROC} data={DATA}")
log("cpu flags: " + " ".join(env["cpu_flags"].split()))
log("gpu: " + sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", 60)[1].strip())
rc, o = sh("apt-get install -y -qq libvulkan1 vulkan-tools > /dev/null 2>&1; vulkaninfo --summary 2>&1 | head -40", 600)
env["vulkan"] = o
log("vulkan probe: " + ("GPU listed" if "NVIDIA" in o else "no NVIDIA Vulkan device"))

sh(f"{sys.executable} -m pip install -q uv", 600)
UV = "uv"


def venv(name, pkgs, extra=""):
    path = VENVS / name
    t = time.time()
    rc, o = sh(f"{UV} venv -q --python 3.12 {path} && {UV} pip install -q --python {path}/bin/python {pkgs} {extra}", 3600)
    log(f"venv {name}: rc={rc} in {time.time() - t:.0f}s")
    if rc:
        (OUT / "logs" / f"venv_{name}.log").write_text(o[-20000:])
    return f"{path}/bin/python" if rc == 0 else None


COMMON = "numpy soundfile psutil nvidia-ml-py huggingface_hub"


def run(py, backend, system, device, sset, threads=NPROC, timeout=5400, extra_env=None):
    if py is None:
        status(backend=backend, system=system, device=device, set=sset, ok=False, error="environment failed")
        return False
    tag = f"{system}__{device}_{sset}"
    out = OUT / "raw" / system / f"{device}_{sset}{'_t' + str(threads) if sset == 'speed' else ''}.jsonl"
    e = dict(os.environ, **(extra_env or {}))
    t = time.time()
    try:
        rc, o = sh(f"{py} -u {BENCH} --backend {backend} --system {system} --device {device} --set {sset} "
                   f"--threads {threads} --data {DATA} --out {out}", timeout, e)
    except subprocess.TimeoutExpired:
        rc, o = -9, "timeout"
    (OUT / "logs" / f"{tag}.log").write_text(o[-30000:])
    ok = rc == 0
    status(backend=backend, system=system, device=device, set=sset, threads=threads, ok=ok, rc=rc,
           seconds=round(time.time() - t), error=None if ok else o.strip().splitlines()[-1][:400] if o.strip() else "")
    log(f"{'OK  ' if ok else 'FAIL'} {backend:6s} {tag} ({time.time() - t:.0f}s)")
    return ok


# ---- 2. Parakeet Redux (Photon): CPU fast path check, CPU run, CUDA attempt -----------------------------------
py = venv("photon", f"moondream==2.6.1 {COMMON}")
if py:
    rc, o = sh(f"{py} -c \"from kestrel_kernels import ternary as t; c=t._cpu; "
               f"print('ternary_gemm_isa', c.ternary_gemm_isa(), {{i: c.gemm_isa_available(i) for i in "
               f"['avx512vnni','avxvnni','avx2']}})\"", 600)
    env["photon_isa"] = o.strip()[-500:]
    log("photon isa: " + env["photon_isa"].splitlines()[-1] if env["photon_isa"] else "photon isa: ?")
for sset in ("accuracy", "speed"):
    run(py, "photon", "redux", "cpu", sset)
for sset in ("accuracy", "speed"):
    if not run(py, "photon", "redux", "gpu", sset):
        break

# ---- 3. Whisper on CUDA (faster-whisper / CTranslate2) --------------------------------------------------------
py = venv("fw", f"faster-whisper 'nvidia-cublas-cu12' 'nvidia-cudnn-cu12==9.*' {COMMON}")
fw_env = None
if py:
    libs = glob.glob(str(VENVS / "fw/lib/python3.12/site-packages/nvidia/*/lib"))
    fw_env = {"LD_LIBRARY_PATH": ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])}
for system in ("wturbo_fw", "wlv3_fw"):
    for sset in ("accuracy", "speed"):
        run(py, "fw", system, "gpu", sset, extra_env=fw_env)

# ---- 4. transcribe-cpp: same engine + GGUF files as the laptop -------------------------------------------------
py = venv("tcpp", f"transcribe-cpp==0.3.1 {COMMON}")
gpu_ok = False
if py:
    rc, o = sh(f"{py} -c \"import transcribe_cpp as t; print([d.device_type for d in t.backends()])\"", 600)
    env["tcpp_backends"] = o.strip()[-2000:]
    gpu_ok = "'gpu'" in o
    log(f"tcpp backends: {o.strip().splitlines()[-1] if o.strip() else '?'}")
run(py, "tcpp", "unified_q8", "cpu", "accuracy")  # cross-platform determinism check vs the laptop
for system in ("tdt_q8", "unified_q8", "tdt_f16", "moon_f16"):
    run(py, "tcpp", system, "cpu", "speed")
if gpu_ok:
    for system in ("tdt_f16", "tdt_q8", "unified_q8", "wlv3_q8", "wturbo_q8", "moon_f16"):
        for sset in ("accuracy", "speed"):
            run(py, "tcpp", system, "gpu", sset)

# ---- 5. Moonshine on CUDA (transformers, Kaggle's preinstalled torch) -----------------------------------------
sh(f"{sys.executable} -m pip install -q nvidia-ml-py psutil soundfile", 900)
for sset in ("accuracy", "speed"):
    run(sys.executable, "hf", "moon_hf", "gpu", sset)

# ---- 6. Parakeet on CUDA with NVIDIA NeMo (original weights) --------------------------------------------------
t = time.time()
rc, o = sh(f"{sys.executable} -m pip install -q 'nemo_toolkit[asr]>=2.7'", 3600)
log(f"nemo install rc={rc} in {time.time() - t:.0f}s")
if rc:
    (OUT / "logs" / "nemo_install.log").write_text(o[-20000:])
for system in ("unified_nemo", "tdt_nemo"):
    for sset in ("accuracy", "speed"):
        run(sys.executable if rc == 0 else None, "nemo", system, "gpu", sset)

(OUT / "env.json").write_text(json.dumps(env, indent=1))
log("ALL DONE")
