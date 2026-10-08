"""Phase 0 probe: can Photon run Parakeet Redux on this CPU if the AVX2 path is forced?"""
import sys, time
from kestrel_kernels import ternary as t
isa = sys.argv[1] if len(sys.argv) > 1 else None
if isa:
    t.set_gemm_isa(isa)
print("isa:", t.ternary_gemm_isa(), flush=True)
import moondream as md
t0 = time.time()
with md.photon("moondream/parakeet-redux", device="cpu") as sp:
    print("load_s", round(time.time() - t0, 2), flush=True)
    for i in range(3):
        a = time.time()
        r = sp.transcribe(audio="data/smoke/jfk.wav")
        print("run", i, round(time.time() - a, 3), "s ->", r["text"], flush=True)
