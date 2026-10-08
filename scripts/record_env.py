"""Write results/env.json: hardware, OS, driver, package versions, model files with SHA-256 (PROTOCOL 6.7)."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from models import TCPP, gguf, HUB  # noqa: E402


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {type(e).__name__}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def pkgs(env, names):
    out = sh(["uv", "pip", "list", "--python", str(ROOT / f"envs/{env}/Scripts/python.exe"), "--format", "json"])
    try:
        return {p["name"]: p["version"] for p in json.loads(out) if p["name"].lower() in names}
    except json.JSONDecodeError:
        return out


env = {
    "os": platform.platform(),
    "cpu": sh(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"]),
    "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    "python": sys.version.split()[0],
    "docker_image": sh(["docker", "image", "inspect", "stt-photon:1", "--format", "{{.Id}}"]),
    "docker_cpus_mem": sh(["docker", "info", "--format", "{{.NCPU}} CPUs, {{.MemTotal}} bytes"]),
    "packages": {
        "tcpp": pkgs("tcpp", {"transcribe-cpp", "transcribe-cpp-native", "numpy", "psutil", "nvidia-ml-py"}),
        "photon_windows": pkgs("photon", {"moondream", "kestrel", "kestrel-kernels", "torch"}),
        "cloud_and_scoring": pkgs("cloud", {"jiwer", "whisper-normalizer", "silero-vad", "yt-dlp", "kaggle",
                                            "requests", "datasets"}),
    },
    "models": {},
}
for sid, (repo, f) in TCPP.items():
    p = gguf(repo, f)
    env["models"][sid] = {"file": f, "repo": f"handy-computer/{repo}", "bytes": Path(p).stat().st_size,
                          "sha256": sha256(p)}
redux = sorted(Path(HUB).glob("models--moondream--parakeet-redux/snapshots/*/*"))
env["models"]["redux"] = {"repo": "moondream/parakeet-redux",
                          "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in redux if p.is_file()}}
(ROOT / "results/env.json").write_text(json.dumps(env, indent=2))
print(json.dumps({k: v for k, v in env.items() if k != "models"}, indent=1)[:1500])
