"""Generate the documented Kaggle notebook kaggle/notebook/stt-benchmark.ipynb (pushed as stt-benchmark-gpu)."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "notebook"
OUT.mkdir(exist_ok=True)
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# Parakeet Redux on a GPU and a server CPU: a controlled speech-to-text benchmark

**Subject:** [Parakeet Redux](https://huggingface.co/moondream/parakeet-redux), Moondream's 178 MB ternary-weight compression of NVIDIA's Parakeet TDT 0.6B v3. The question: **how does it perform, and is it useful in real-world scenarios?** The other models are reference points.

This notebook is the second run of a benchmark first done on a Windows laptop (Ryzen 5 5600H, GTX 1650). That run could not test three things, which this one can:

1. **Does Redux run on a GPU?** Its Photon engine officially needs an NVIDIA Ampere-or-newer GPU. Kaggle's T4 is Turing, so this tests outside official support.
2. **Does a server CPU change its speed?** Photon has faster kernels for AVX-VNNI and AVX-512-VNNI.
3. **How do the reference models perform in their standard CUDA runtimes?** NVIDIA NeMo for Parakeet, faster-whisper for Whisper, transformers for Moonshine.

| Input | Contents |
| --- | --- |
| `stt-benchmark-audio` | The 147 audio units (35.38 min of English) every system receives |
| `stt-benchmark-code` | Worker (`bench.py`), scorer, human reference transcripts |

Author: Abhash Chakraborty, October 2026. Both datasets are private: they contain copyrighted clips (research use) and a private voice recording.
""")

md(r"""
## Method

**Test data (5 subsets).** Public subsets use provided captions/transcripts or the fixed accent passage; private dictation has both script and model-consensus-adjusted references.

| Subset | Content |
| --- | --- |
| A: real-world | 6 YouTube clips with human caption tracks: TED, podcast interview, MIT lecture, PBS NewsHour, TEDx (Indian-English speaker), street interviews |
| B: benchmark | 50 Open ASR Leaderboard utterances (LibriSpeech clean and other, Earnings-22, AMI, VoxPopuli) |
| B2: accents | 12 Speech Accent Archive speakers of 12 first languages, one fixed passage |
| C: noise | The 10 LibriSpeech-clean clips with pink noise at 5 dB SNR |
| D: dictation | The author dictating 4 paragraphs into a laptop microphone |

**Fairness.**
- Every system gets byte-identical audio units.
- Each configuration (system, device, set) runs in a **fresh process** with its own environment.
- Decoding is greedy, English where the API allows, no prompts.

**Metrics.**
- **WER:** word error rate after the Whisper English text normalizer (Open ASR Leaderboard convention), with 95% bootstrap confidence intervals and paired bootstrap tests.
- **RTFx:** audio seconds processed per second (higher is faster), the median of 3 timed passes after a warm-up pass over a 20-clip speed subset.
- **Latency:** per-clip p50 and p95.
- **Resources:** cold start, peak RAM, peak GPU memory above idle.

**Sets.** `accuracy` = all 147 units, one pass. `speed` = 20 benchmark clips (180.289 seconds, 3.00 min), 1 warm-up + 3 timed passes.

**Limitations.** The dictation reference called `D_verbatim` was adjusted using model consensus; it is not independent human-verified ground truth. Script scores are retained too. Ten noise clips derive from clean clips already in the public set, so clip-bootstrap CIs do not capture that dependency. Confidence intervals are exploratory; overlapping intervals do not prove equivalence. Different runtimes and precisions confound a pure compression comparison. Cold starts can include uncached downloads.
""")

code(r"""
import glob, json, os, subprocess, sys, time
from pathlib import Path

AUDIO = str(Path(glob.glob('/kaggle/input/**/stt-benchmark-audio/**/units.csv', recursive=True)[0]).parent)
CODE = str(Path(glob.glob('/kaggle/input/**/stt-benchmark-code/**/bench.py', recursive=True)[0]).parent)
BENCH = f'{CODE}/bench.py'
OUT = Path('/kaggle/working/out'); (OUT / 'logs').mkdir(parents=True, exist_ok=True)
VENVS = Path('/kaggle/working/venvs')
NPROC = os.cpu_count()
T0 = time.time()

def log(msg):
    line = f'[{time.time() - T0:6.0f}s] {msg}'
    print(line, flush=True)
    with open(OUT / 'main.log', 'a') as f: f.write(line + '\n')

def sh(cmd, timeout=3600, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, (r.stdout or '') + (r.stderr or '')

def venv(name, pkgs):
    path = VENVS / name
    rc, o = sh(f'uv venv -q --python 3.12 {path} && uv pip install -q --python {path}/bin/python {pkgs}', 3600)
    log(f'environment {name}: {"ready" if rc == 0 else "FAILED"}')
    if rc: (OUT / 'logs' / f'venv_{name}.log').write_text(o[-20000:])
    return f'{path}/bin/python' if rc == 0 else None

def run(py, backend, system, device, sset, extra_env=None, timeout=5400):
    # One configuration in a fresh process; failures are logged and never stop the notebook.
    tag = f'{system}__{device}_{sset}'
    out = OUT / 'raw' / system / f'{device}_{sset}{"_t" + str(NPROC) if sset == "speed" else ""}.jsonl'
    t = time.time()
    if py is None:
        rc, o = 1, 'environment failed'
    else:
        try:
            rc, o = sh(f'{py} -u {BENCH} --backend {backend} --system {system} --device {device} --set {sset} '
                       f'--threads {NPROC} --data {AUDIO} --out {out}', timeout, dict(os.environ, **(extra_env or {})))
        except subprocess.TimeoutExpired:
            rc, o = -9, 'timeout'
    (OUT / 'logs' / f'{tag}.log').write_text(o[-30000:])
    ok = rc == 0
    with open(OUT / 'status.jsonl', 'a') as f:
        f.write(json.dumps({'backend': backend, 'system': system, 'device': device, 'set': sset, 'ok': ok,
                            'seconds': round(time.time() - t),
                            'error': None if ok else (o.strip().splitlines() or [''])[-1][:300]}) + '\n')
    log(f'{"OK  " if ok else "FAIL"} {backend:6s} {tag} ({time.time() - t:.0f}s)')
    return ok

sh(f'{sys.executable} -m pip install -q uv', 600)
COMMON = 'numpy soundfile psutil nvidia-ml-py huggingface_hub'
print('audio:', AUDIO); print('code:', CODE)
""")

md(r"""
## Hardware

The cell below records the machine: CPU model and instruction sets (Photon's fastest CPU kernels need `avx512_vnni` or `avx_vnni`), the GPU, and whether a Vulkan driver is present (the transcribe-cpp engine used on the laptop runs on GPUs only through Vulkan).
""")

code(r"""
env = {'nproc': NPROC}
for name, cmd in [('lscpu', 'lscpu'), ('nvidia_smi', 'nvidia-smi'), ('free', 'free -m'),
                  ('cpu_flags', "grep -m1 -o -w -E 'avx2|avx512f|avx512_vnni|avx_vnni|amx_tile' /proc/cpuinfo | sort -u")]:
    env[name] = sh(cmd, 60)[1]
print([l.strip() for l in env['lscpu'].splitlines() if 'Model name' in l][0])
print('logical CPUs:', NPROC, '| relevant CPU flags:', ' '.join(env['cpu_flags'].split()))
print(sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader', 60)[1].strip())
rc, o = sh('apt-get install -y -qq libvulkan1 vulkan-tools > /dev/null 2>&1; vulkaninfo --summary 2>&1 | head -40', 600)
env['vulkan'] = o
print('Vulkan GPU available:', 'NVIDIA' in o)
""")

md(r"""
## Stage 1: Parakeet Redux (the subject)

Photon picks its CPU kernel at load time. The cell first prints which kernel this CPU gets, then runs Redux:
1. on the **CPU**, with all logical CPUs;
2. on the **T4 GPU**, outside Photon's official Ampere-or-newer support.
""")

code(r"""
py = venv('photon', f'moondream==2.6.1 {COMMON}')
if py:
    rc, o = sh(f"{py} -c \"from kestrel_kernels import ternary as t; c=t._cpu; print('kernel:', c.ternary_gemm_isa(), "
               f"{{i: c.gemm_isa_available(i) for i in ['avx512vnni','avxvnni','avx2']}})\"", 600)
    env['photon_isa'] = o.strip()[-400:]
    print(env['photon_isa'].splitlines()[-1])
for sset in ('accuracy', 'speed'):
    run(py, 'photon', 'redux', 'cpu', sset)
for sset in ('accuracy', 'speed'):
    run(py, 'photon', 'redux', 'gpu', sset)
""")

md(r"""
## Stage 2: Whisper large-v3 and large-v3-turbo (faster-whisper, CUDA float16)

Whisper large-v3 (1.55B parameters) is the large open reference; turbo (0.8B) is its faster variant. Greedy decoding (`beam_size=1`), English, no VAD filter, no conditioning on previous text.
""")

code(r"""
py = venv('fw', f"faster-whisper 'nvidia-cublas-cu12' 'nvidia-cudnn-cu12==9.*' {COMMON}")
fw_env = {'LD_LIBRARY_PATH': ':'.join(glob.glob(str(VENVS / 'fw/lib/python3.12/site-packages/nvidia/*/lib'))
                                      + [os.environ.get('LD_LIBRARY_PATH', '')])}
for system in ('wturbo_fw', 'wlv3_fw'):
    for sset in ('accuracy', 'speed'):
        run(py, 'fw', system, 'gpu', sset, fw_env)
""")

md(r"""
## Stage 3: Moonshine base (transformers, CUDA float16)

Moonshine base (61M parameters, 132 MB) is the edge-model reference, close to Redux's file size.
""")

code(r"""
sh(f'{sys.executable} -m pip install -q nvidia-ml-py psutil soundfile', 900)
for sset in ('accuracy', 'speed'):
    run(sys.executable, 'hf', 'moon_hf', 'gpu', sset)
""")

md(r"""
## Stage 4: the original model, in NVIDIA NeMo

The fairest reference for Redux is the model it was compressed from (installed after Moonshine because NeMo changes packages in Kaggle's base environment), **Parakeet TDT 0.6B v3**, with its original weights, in NVIDIA's own runtime, on the same GPU. **Parakeet Unified EN 0.6B** (an English-only, streaming-capable sibling) runs here too.
""")

code(r"""
rc, o = sh(f"{sys.executable} -m pip install -q nvidia-ml-py psutil soundfile 'nemo_toolkit[asr]>=2.7'", 3600)
log(f'NeMo install: {"ok" if rc == 0 else "FAILED"}')
for system in ('tdt_nemo', 'unified_nemo'):
    for sset in ('accuracy', 'speed'):
        run(sys.executable if rc == 0 else None, 'nemo', system, 'gpu', sset)
""")

md(r"""
## Stage 5: the laptop's exact engine and files on this CPU (transcribe-cpp, GGUF)

These are the same GGUF files and engine the laptop used: 8-bit and 16-bit Parakeet TDT, Parakeet Unified, Moonshine. They run on this CPU for a like-for-like CPU comparison with Redux. If this machine exposed a Vulkan GPU, they would also run on the GPU. The Parakeet Unified accuracy pass checks that results reproduce across machines.
""")

code(r"""
py = venv('tcpp', f'transcribe-cpp==0.3.1 {COMMON}')
gpu_ok = False
if py:
    rc, o = sh(f'{py} -c "import transcribe_cpp as t; print([d.device_type for d in t.backends()])"', 600)
    gpu_ok = "'gpu'" in o
    print('transcribe-cpp devices:', o.strip().splitlines()[-1] if o.strip() else '?')
run(py, 'tcpp', 'unified_q8', 'cpu', 'accuracy')
for system in ('tdt_q8', 'unified_q8', 'tdt_f16', 'moon_f16'):
    run(py, 'tcpp', system, 'cpu', 'speed')
if gpu_ok:
    for system in ('tdt_f16', 'tdt_q8', 'unified_q8', 'moon_f16'):
        for sset in ('accuracy', 'speed'):
            run(py, 'tcpp', system, 'gpu', sset)
(OUT / 'env.json').write_text(json.dumps(env, indent=1))
""")

md(r"""
## Scoring

The scorer is the same one used for the laptop run (`scripts/score.py`, `scripts/score_speed.py` in `stt-benchmark-code`). Results are written to `/kaggle/working/out` (`metrics_accuracy.csv`, `metrics_speed.csv`, `per_clip.csv`, `paired.csv`).
""")

code(r"""
import shutil
REPO = Path('/kaggle/working/repo')
shutil.rmtree(REPO, ignore_errors=True)
shutil.copytree(CODE, REPO)
(REPO / 'kaggle').mkdir(exist_ok=True)
os.symlink(OUT, REPO / 'kaggle' / 'results')
sh(f'{sys.executable} -m pip install -q jiwer whisper-normalizer pandas', 900)
for script in ('score.py', 'score_speed.py'):
    rc, o = sh(f'cd {REPO} && STT_RESULTS=kaggle/results {sys.executable} scripts/{script}', 1800)
    print(script, 'ok' if rc == 0 else 'FAILED'); print(o[-3000:])
""")

code(r"""
import pandas as pd
pd.set_option('display.width', 200)
names = {'redux': 'Parakeet Redux (178 MB)', 'tdt_nemo': 'Parakeet TDT 0.6B v3, original (NeMo)',
         'unified_nemo': 'Parakeet Unified EN 0.6B (NeMo)', 'unified_q8': 'Parakeet Unified EN 0.6B (GGUF 8-bit)',
         'wlv3_fw': 'Whisper large-v3', 'wturbo_fw': 'Whisper large-v3-turbo', 'moon_hf': 'Moonshine base'}
m = pd.read_csv(OUT / 'metrics_accuracy.csv')
acc = m[m.subset.isin(['ALL_PUBLIC', 'A', 'B', 'B2', 'C', 'D_verbatim'])].pivot_table(
    index=['system', 'device'], columns='subset', values='wer') * 100
acc = acc.sort_values('ALL_PUBLIC').round(2)
acc.index = [f'{names.get(s, s)} [{d}]' for s, d in acc.index]
print('Word error rate (%), lower is better. ALL_PUBLIC = subsets A+B+B2+C (78 clips, 5,531 words).')
acc
""")

code(r"""
import numpy as np
p = pd.read_csv(OUT / 'per_clip.csv'); p = p[~p.subset.str.startswith('D')]
rng = np.random.default_rng(42)
rows = []
def paired(a, da, b, db):
    A = p[(p.system == a) & (p.device == da)].set_index('clip'); B = p[(p.system == b) & (p.device == db)].set_index('clip')
    ids = A.index.intersection(B.index)
    assert len(ids) == len(A) == len(B) == 78, 'Incomplete paired public comparison'
    rng = np.random.default_rng(42)
    ea, eb, n = A.loc[ids, 'err'].values, B.loc[ids, 'err'].values, A.loc[ids, 'n'].values
    idx = rng.integers(0, len(ids), (10000, len(ids))); d = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1) * 100
    rows.append({'comparison': f'{names.get(a, a)} [{da}] minus {names.get(b, b)} [{db}]', 'clips': len(ids),
                 'diff_points': round(100 * (ea.sum() - eb.sum()) / n.sum(), 2),
                 'ci95_low': round(np.percentile(d, 2.5), 2), 'ci95_high': round(np.percentile(d, 97.5), 2),
                 'share_of_resamples_worse': round(float((d > 0).mean()), 3)})
paired('redux', 'gpu', 'tdt_nemo', 'gpu'); paired('redux', 'cpu', 'tdt_nemo', 'gpu')
paired('redux', 'gpu', 'unified_nemo', 'gpu'); paired('redux', 'gpu', 'wturbo_fw', 'gpu')
paired('redux', 'gpu', 'unified_q8', 'cpu'); paired('redux', 'cpu', 'unified_nemo', 'gpu')
paired('redux', 'cpu', 'wturbo_fw', 'gpu'); paired('redux', 'cpu', 'unified_q8', 'cpu')
print('Paired bootstrap on identical clips (10,000 resamples). Positive = Redux makes more errors.')
paired_results = pd.DataFrame(rows)
paired_results.to_csv(OUT / 'paired_kaggle.csv', index=False)
paired_results
""")

code(r"""
s = pd.read_csv(OUT / 'metrics_speed.csv')
s['system'] = [f'{names.get(a, a)} [{d}]' for a, d in zip(s.system, s.device)]
s['minutes_per_hour_of_audio'] = (60 / s.rtfx).round(2)
cols = ['system', 'rtfx', 'minutes_per_hour_of_audio', 'latency_p50_s', 'latency_p95_s', 'cold_start_s',
        'peak_rss_mb', 'peak_vram_delta_mb']
print('Speed: RTFx = audio seconds per processing second (higher is faster); latency per 4-25 s clip.')
s[cols].sort_values('rtfx', ascending=False).round(2).reset_index(drop=True)
""")

code(r"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
a = m[m.subset == 'ALL_PUBLIC'].copy(); a['label'] = [f'{names.get(x, x)} [{d}]' for x, d in zip(a.system, a.device)]
a = a.sort_values('wer', ascending=False)
col = ['#2f6fd6' if x == 'redux' else '#9a9a92' for x in a.system]
axes[0].errorbar(a.wer * 100, range(len(a)), xerr=[(a.wer - a.ci_lo) * 100, (a.ci_hi - a.wer) * 100], fmt='none',
                 ecolor=col, elinewidth=2)
axes[0].scatter(a.wer * 100, range(len(a)), c=col, s=60, zorder=3)
axes[0].set_yticks(range(len(a)), a.label); axes[0].set_xlabel('WER % (95% CI), lower is better')
axes[0].set_title('Accuracy, 78 public clips', loc='left', fontweight='bold')
sp = s.sort_values('rtfx'); colr = ['#2f6fd6' if 'Redux' in x else '#9a9a92' for x in sp.system]
axes[1].barh(sp.system, sp.rtfx, color=colr)
for i, v in enumerate(sp.rtfx): axes[1].text(v + 0.5, i, f'{v:.1f}x', va='center')
axes[1].set_xlabel('Real-time factor (higher is faster)'); axes[1].set_title('Speed', loc='left', fontweight='bold')
for ax in axes: ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / 'summary.png', dpi=150); plt.show()
""")

md(r"""
## How to read the results

- **Accuracy:** assess the explicit paired interval. Overlapping separate intervals do not establish equivalence, and a fixed "under one point is noise" rule is not justified. These intervals also omit the clean/noisy-clip dependency.
- **Redux vs its original:** the key row in the paired table is *Parakeet Redux [gpu] minus Parakeet TDT 0.6B v3, original [gpu]*. On the laptop the same comparison was +0.80 points.
- **GPU memory:** Photon's documentation states CUDA runs Redux dense (uncompressed), so its 178 MB file size does not mean small GPU memory. Check `peak_vram_delta_mb`.
- **Redux output is not bit-identical across machines or devices:** the saved CPU and GPU WER differ, so device equivalence cannot be assumed.

The full write-up, including the real-world verdict by scenario, is in the accompanying report.
""")

code(r"""
# Keep the notebook output small: drop the environments, keep results and logs.
shutil.rmtree(VENVS, ignore_errors=True)
shutil.rmtree(REPO, ignore_errors=True)
print(sorted(p.name for p in OUT.iterdir()))
log('ALL DONE')
""")

# Default to analysis of verified measurements. Re-running every model is an explicit opt-in.
cells.insert(1, {"cell_type": "markdown", "metadata": {}, "source": [
    "## Execution mode and provenance\n\n",
    "The previous documented notebook ended in ERROR before emitting execution logs; Kaggle provided no failure reason. ",
    "This version defaults to **reproducing the analysis of completed measurements**, without using GPU quota for new inference. ",
    "Results combine the original T4 script session (13 successful configurations) with the completed repair session ",
    "(6 configurations). This is **not a new single-session benchmark**. Raw files, original failure logs and SHA-256 ",
    "provenance are retained. Set `RUN_BENCHMARK = True` below and enable a GPU to execute the five inference stages instead.\n\n",
    "The measurement CPU had AVX-512F but **no VNNI** and Photon selected AVX2. ",
    "This does not test the advertised AVX-512-VNNI fast path.\n"
]})
for cell in cells:
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    if src.startswith('import glob, json, os, subprocess'):
        src = src.replace('NPROC = os.cpu_count()', 'RUN_BENCHMARK = False\nNPROC = 4')
        src = src.replace("sh(f'{sys.executable} -m pip install -q uv', 600)",
                          "if RUN_BENCHMARK: sh(f'{sys.executable} -m pip install -q uv', 600)")
        src += '''\nif not RUN_BENCHMARK:
    import shutil
    shutil.copytree(Path(CODE) / 'recovered_results', OUT, dirs_exist_ok=True)
    log('ANALYSIS REPLAY: original plus repair measurements; no new inference')
'''
    elif src.startswith("env = {'nproc': NPROC}"):
        src = "if RUN_BENCHMARK:\n" + ''.join('    ' + line + '\n' for line in src.splitlines())
        src += "else:\n    env = json.loads((OUT / 'env.json').read_text())\n    print('MEASUREMENT hardware:', env['nproc'], 'logical CPUs; Photon:', env['photon_isa'])\n    print(env['nvidia_smi'])\n"
    elif (src.startswith("py = venv(") or src.startswith("sh(f'{sys.executable} -m pip install -q nvidia-ml-py")
          or src.startswith('rc, o = sh(f"{sys.executable} -m pip install -q nvidia-ml-py')):
        src = "if RUN_BENCHMARK:\n" + ''.join('    ' + line + '\n' for line in src.splitlines())
        src += "else:\n    print('Using validated saved measurements for this stage.')\n"
    elif src.startswith('import shutil\nREPO'):
        src = src.replace("    print(script, 'ok' if rc == 0 else 'FAILED'); print(o[-3000:])",
                          "    print(script, 'ok' if rc == 0 else 'FAILED'); print(o[-3000:])\n    if rc: raise RuntimeError(f'{script} failed; cannot publish partial scores')")
        src = '''import sys
sys.path.insert(0, CODE)
from validate_results import validate
provenance = json.loads((OUT / 'provenance.json').read_text()) if not RUN_BENCHMARK else None
hashes = {f['path']: f['sha256'] for f in provenance['files']} if provenance else None
validation = validate(OUT, Path(CODE) / 'data/units.csv', hashes)
(OUT / 'validation.json').write_text(json.dumps(validation, indent=2))
print('Validation:', {k: v for k, v in validation.items() if k != 'files'})
''' + src
    elif src.startswith('import numpy as np\np = pd.read_csv'):
        src = src.replace('pd.DataFrame(rows)\n', "paired_results = pd.DataFrame(rows)\npaired_results.to_csv(OUT / 'paired_kaggle.csv', index=False)\npaired_results\n")
    cell['source'] = src.splitlines(keepends=True)

for i, cell in enumerate(cells): cell['id'] = f'stt-{i:02d}'
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(OUT / "stt-benchmark.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "abhashchakraborty/stt-benchmark-gpu", "title": "stt-benchmark-gpu", "code_file": "stt-benchmark.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": ["abhashchakraborty/stt-benchmark-audio", "abhashchakraborty/stt-benchmark-code"],
    "competition_sources": [], "kernel_sources": []}, indent=1))
print("cells:", len(cells))
