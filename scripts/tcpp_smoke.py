"""Phase 0 smoke test: each GGUF model on CPU (6 threads) and on the GTX 1650 via Vulkan."""
import sys, time, glob, os
import numpy as np, soundfile as sf
import transcribe_cpp as tc

HUB = os.path.expanduser("~/.cache/huggingface/hub")
pcm, sr = sf.read("data/smoke/jfk.wav", dtype="float32")
assert sr == 16000 and pcm.ndim == 1
devs = {d.device_type if d.device_type != "gpu" else "gpu": d for d in tc.backends()}
for pattern in sys.argv[1:]:
    paths = glob.glob(f"{HUB}/models--handy-computer--*/snapshots/*/{pattern}")
    if not paths:
        print("MISSING", pattern); continue
    for dev in ("cpu", "gpu"):
        try:
            t0 = time.time()
            with tc.Model(paths[0], device=devs[dev]) as m:
                load = time.time() - t0
                with m.session(n_threads=6) as s:
                    runs = []
                    for _ in range(3):
                        a = time.time(); r = s.run(pcm, language="en"); runs.append(time.time() - a)
            print(f"{pattern:40s} {dev}: load {load:5.2f}s  runs {['%.3f' % x for x in runs]}  -> {r.text!r}", flush=True)
        except Exception as e:
            print(f"{pattern:40s} {dev}: FAIL {type(e).__name__}: {e}", flush=True)
