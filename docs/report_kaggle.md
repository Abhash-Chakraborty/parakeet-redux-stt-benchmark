# Parakeet Redux: Is a 178 MB Speech Model Useful in the Real World?

Abhash Chakraborty · 6 October 2026 · Final Kaggle supplement

**Verdict:** Redux is a practical option when a small model download and fast CPU transcription matter. It is also fast on the tested T4 GPU, but gives up some accuracy versus its original model. Its 178 MB weights do not imply a tiny runtime. The evidence supports a specific trade-off, not a universal winner.

## What was completed

The original Kaggle script produced 13 successful configurations. The separate repair script completed the 6 configurations affected by the PyAV and NeMo configuration bugs. The documented notebook then ended in `ERROR`, with no execution log or failure explanation exposed by Kaggle's API. No cause is inferred from that empty log.

The finalized measurements therefore combine **two completed inference sessions**, both on T4 hardware. They are not described as a clean single-session rerun. The recovered raw files are byte-identical to their archived/downloaded sources; `provenance.json` lists their SHA-256 hashes and source sessions. `validation.json` checks 8 accuracy configurations × 147 units, plus 11 speed configurations × 20 units × 3 passes, exact unit identities, durations, duplicate records, positive timings and completion records. Original failures remain in `sessions/`.

The final documented notebook **v5 completed successfully on Kaggle**. Its exported accuracy/speed scores agree with local scoring, all source provenance matches, and all eight explicit paired comparisons were saved. It defaults to validating and rescoring these saved measurements and performs no new inference in this mode. `RUN_BENCHMARK = True` opts into the original five inference stages; a GPU must also be enabled. That optional mode has not been newly executed in this completion pass.

## Hardware and method

The measurement host was an Intel Xeon at 2.00 GHz, 2 cores / 4 logical CPUs, 32 GB RAM and NVIDIA Tesla T4 (15,360 MiB), Ubuntu 24.04.1. Both sessions logged T4 and driver 580.178.04. Workers used 4 threads. The first session recorded AVX2 and AVX-512F but **no AVX-VNNI or AVX-512-VNNI**; Photon selected its AVX2 kernel. This is not a test of the advertised VNNI fast path. No Vulkan GPU driver was available, so GGUF references ran on the CPU.

All systems use the same 147 pre-segmented English audio units. Public accuracy covers 78 parent clips and 5,531 reference words: six YouTube clips, 50 benchmark utterances, 12 accent passages and 10 noisy derivatives. Four private dictated paragraphs are reported separately. Each speed configuration uses the same 20 benchmark clips, 180.29 seconds of audio, a warm-up and three timed passes. RTFx is the median throughput; latency is per-clip p50/p95. RAM and GPU-memory deltas were sampled every 50 ms.

The same `scripts/score.py` normalizer and scorer as the laptop experiment were used. The supplemental paired comparisons below explicitly match system **and device**, with 10,000 resamples and seed 42. Different runtimes and precisions remain confounders: Photon vs NeMo is a practical system comparison and cannot isolate compression alone.

## Accuracy

| System | Device | WER | 95% clip bootstrap CI |
| --- | --- | --- | --- |
| Parakeet Unified Q8 (GGUF) | cpu | 4.14% | 2.59–5.52% |
| Parakeet Unified (NeMo) | gpu | 4.25% | 2.69–5.60% |
| Whisper large-v3-turbo (CUDA) | gpu | 4.56% | 3.10–5.93% |
| Parakeet TDT v3 (NeMo) | gpu | 4.72% | 2.60–6.58% |
| Whisper large-v3 (CUDA) | gpu | 4.88% | 3.13–6.43% |
| Parakeet Redux | gpu | 5.32% | 3.47–7.11% |
| Parakeet Redux | cpu | 5.50% | 3.47–7.47% |
| Moonshine base (CUDA) | gpu | 7.07% | 5.29–8.95% |

| System/device | YouTube A | Benchmark B | Accents B2 | Noise C | Dictation script | Dictation consensus |
| --- | --- | --- | --- | --- | --- | --- |
| Parakeet Redux cpu | 6.49% | 4.40% | 3.86% | 1.82% | 11.45% | 4.89% |
| Parakeet Redux gpu | 6.17% | 4.61% | 3.62% | 2.18% | 11.89% | 5.33% |
| Parakeet TDT v3 (NeMo) gpu | 5.80% | 4.20% | 2.05% | 1.09% | 9.25% | 2.67% |

Redux GPU scores **5.32% WER**, versus **4.72%** for the original Parakeet in NeMo. CPU Redux scores 5.50%. Device and runtime can change transcripts; the laptop's 10-clip device check does not establish universal CPU/GPU equivalence.

| Comparison (Redux minus reference) | WER difference | 95% CI |
| --- | --- | --- |
| Redux gpu – Parakeet TDT v3 (NeMo) gpu | +0.60 points | -0.25 to +1.26 |
| Redux gpu – Parakeet Unified (NeMo) gpu | +1.07 points | +0.13 to +1.98 |
| Redux gpu – Whisper large-v3-turbo (CUDA) gpu | +0.76 points | -0.65 to +1.75 |
| Redux gpu – Parakeet Unified Q8 (GGUF) cpu | +1.18 points | +0.22 to +2.11 |
| Redux cpu – Parakeet TDT v3 (NeMo) gpu | +0.78 points | -0.08 to +1.50 |
| Redux cpu – Parakeet Unified (NeMo) gpu | +1.25 points | +0.19 to +2.28 |
| Redux cpu – Whisper large-v3-turbo (CUDA) gpu | +0.94 points | -0.62 to +2.09 |
| Redux cpu – Parakeet Unified Q8 (GGUF) cpu | +1.36 points | +0.30 to +2.40 |

The central GPU comparison is **+0.60 WER points**, with 95% CI **-0.25 to +1.26**. The interval includes zero, so this sample does not establish a difference at that confidence level. It also exceeds the pre-registered +0.5-point non-inferiority margin: near-equal accuracy is not demonstrated. The fraction of bootstrap resamples with a positive difference is not a posterior probability or a conventional p-value. Overlapping separate confidence intervals do not prove equivalence.

## Speed and resources

| System | Device | RTFx | p50 / p95 (s) | Peak RAM (MiB) | GPU delta (MiB) |
| --- | --- | --- | --- | --- | --- |
| Parakeet Redux | cpu | 7.69x | 0.972 / 2.032 | 1349 | 0 |
| Moonshine base F16 (GGUF) | cpu | 6.77x | 0.834 / 2.986 | 339 | 0 |
| Parakeet TDT F16 (GGUF) | cpu | 6.65x | 1.124 / 2.435 | 2038 | 0 |
| Parakeet Unified Q8 (GGUF) | cpu | 4.94x | 1.550 / 3.281 | 1237 | 0 |
| Parakeet TDT Q8 (GGUF) | cpu | 4.82x | 1.564 / 3.398 | 1367 | 0 |
| Parakeet Redux | gpu | 84.97x | 0.087 / 0.174 | 2370 | 2093 |
| Parakeet TDT v3 (NeMo) | gpu | 73.83x | 0.106 / 0.176 | 2168 | 2875 |
| Parakeet Unified (NeMo) | gpu | 46.13x | 0.162 / 0.305 | 2110 | 5073 |
| Whisper large-v3-turbo (CUDA) | gpu | 25.52x | 0.343 / 0.406 | 587 | 2263 |
| Moonshine base (CUDA) | gpu | 24.29x | 0.302 / 0.636 | 1749 | 311 |
| Whisper large-v3 (CUDA) | gpu | 12.78x | 0.656 / 0.967 | 587 | 4119 |

Redux reaches **84.97x real time on T4**, versus **73.83x** for its original in NeMo (about 15% higher throughput). At that measured rate, one hour of audio takes about 42 seconds. This is sequential warm inference on short clips, not a streaming delay or a measured hour-long job. Redux's GPU memory delta is about **2.04 GiB**, despite its 178 MB weight file. Its CPU peak RAM is about **1.32 GiB** here.

On the Xeon, Redux reaches **7.69x**, compared with **6.65x** for original F16 GGUF and **4.82x** for Q8 GGUF. The server CPU result is slower than the laptop's 15.08x Redux run; the CPUs, thread counts and environments differ. More AVX-512 flags alone do not predict a faster Photon kernel.

Cold-start fields remain available in the CSV. They include environment imports, model initialization and possibly uncached downloads. Cache state was not standardized, so these values must not be ranked as pure model load latency. GPU deltas are device-wide sampled allocations, not exact tensor sizes.

## Laptop versus Kaggle: context from the supplied summary

The laptop could not measure Redux CUDA inference and used Vulkan for the GGUF references. Kaggle added Photon CUDA, the reference models' standard CUDA engines, and a Xeon instruction-path check. It did not reach a VNNI CPU fast path. The laptop had 6 cores / 12 threads (6 threads in the main comparison), 15.3 GB RAM and a GTX 1650; Kaggle used 2 cores / 4 threads, 32 GB RAM and T4 hardware. The weaker measured server CPU and stronger GPU should not be conflated into a single hardware improvement.

| Measure | Laptop | Kaggle |
| --- | --- | --- |
| Redux public WER | 5.33%, CPU | 5.50% CPU; 5.32% GPU |
| Original public WER | 4.54%, F16 GGUF | 4.72%, original weights in NeMo |
| Paired Redux minus original | +0.80 points | +0.60 points, GPU versus GPU |
| Redux CPU throughput | 15.08x | 7.69x |
| Q8 parent CPU throughput | 9.26x native; 10.72x in the same container | 4.82x |
| F16 parent CPU throughput | 7.56x | 6.65x |

The accuracy-gap point estimates are directionally consistent across these deployments; neither establishes equivalence or a significant parent-model gap at the reported 95% level. Runtime, precision, operating system and hardware changes prevent attribution to compression or instruction sets alone. The F16/Q8 speed ranking reverses between machines, demonstrating a deployment-dependent trade-off rather than establishing an instruction-set cause.

Redux was the fastest tested **Parakeet CPU configuration**, not the fastest model overall: Moonshine reached 20.33x on the laptop CPU, versus Redux's 15.08x, with worse accuracy. There was no pre-defined accuracy threshold supporting the phrase "fastest accurate model". CPU per-core scaling was not measured. On T4, Redux's roughly 15% throughput improvement is real in this sample, so "no GPU advantage" is too strong; compact runtime memory and equivalent accuracy were not demonstrated.

| Exact transcript-string comparison | Identical units |
| --- | --- |
| Redux laptop CPU versus Kaggle CPU | 134 / 147 |
| Redux Kaggle CPU versus GPU | 127 / 147 |
| Unified GGUF laptop versus Kaggle CPU | 144 / 147 |

These counts were recomputed directly from the finalized raw JSONL. The supplied summary's Redux CPU/GPU count of 129/147 is corrected to **127/147**; trimming and collapsing whitespace also give 127. Do not infer device-independent transcripts or assume a universal 0.2-point noise threshold.

## Real-world usefulness

| Scenario | Assessment backed by the measurements |
| --- | --- |
| Offline English on a CPU | Useful when download size and throughput matter: 178 MB weights, 15.08x on the laptop and 7.69x on this Xeon. Allow roughly 1–1.4 GiB of RAM for the tested runtime. |
| Accented speech and personal dictation | Use with care. The accent subset and private dictation show extra errors versus the parent model; inspect names and phonetic substitutions. Only 12 accent speakers and one dictation speaker were tested. |
| GPU batch transcription | Promising on this T4: 84.97x with 5.32% public WER. The vendor documents Ampere or newer; success on this Turing card is an empirical observation, not an official support guarantee. |
| Windows on an AVX2-only CPU | The tested Photon Windows wheel failed; Linux on the same laptop worked. This is a version-specific packaging result, not proof that all Windows configurations fail. |
| Tiny-memory edge devices | Not demonstrated. Small weights are attractive but the tested runtime consumes substantially more memory. No phone or embedded board was tested. |
| Highest accuracy among these systems | Redux is not the lowest observed WER. Its original and Unified references have lower point estimates; choose using workload-specific validation rather than size alone. |

## Limits and claims to avoid

- These are small English-only samples on two hardware setups. Speed results use short clips, batch size one and different engines; large production batches, streaming, other languages and current cloud products are outside the Kaggle measurements.
- The 10 noisy clips derive from 10 clean clips already in the public set. Clip bootstrap intervals treat them separately and do not model that dependency. Accent speakers share one passage. The intervals are exploratory, not definitive population-level uncertainty estimates.
- `D_verbatim` is **model-consensus-adjusted**, not independently human-verified verbatim ground truth. It can favor the systems used to build it. Both script and consensus scores remain available; the name-pronunciation question is unresolved and no reference was changed in this completion pass.
- Model revisions and every dependency were not fully pinned or archived in the original sessions. Hashes establish integrity of the recovered files, not reproduction of every upstream model revision. Greedy decoding and thread settings also differ in what each engine exposes.
- No new cloud requests were made. Cloud cost and privacy claims in the older laptop report are historical and require checking current official terms before reuse.
- Do not call 8-bit conversion universally lossless, infer statistical equivalence from interval overlap, claim accuracy is independent of device, or generalize observed T4 success to every unsupported GPU.

## Future scope retained from the supplied summary

1. Test a VNNI-capable CPU and a documented supported GPU, recording kernel selection, precision, memory, cache state and pinned revisions. A newer GPU alone does not demonstrate compressed CUDA inference.
2. Test Apple Silicon and an ARM board. macOS, phones and embedded hardware were not measured here, so their performance remains a proposed experiment.
3. Expand accent coverage substantially (for example 100+ speakers), and evaluate other supported languages with independent references.
4. Revisit native Windows AVX2 support and lighter runtime packaging when concrete releases are available.

These are proposed follow-up studies, not scheduled jobs, promised fixes or completed measurements. The public audio covers **33.20 minutes**; the entire set including private dictation covers **35.38 minutes**. The supplied summary's "78 public clips (36 min)" conflated those totals.

## Reproduction and files

From the repository root, run `envs/cloud/Scripts/python.exe kaggle/finalize.py` to verify hashes and rescore; then run `envs/cloud/Scripts/python.exe kaggle/analyze.py` to regenerate this report, paired comparisons and figure. The laptop's `results/` and archived `kaggle/results_run1/` remain unchanged.

- `kaggle/results/`: validated recovered raw outputs, metrics, explicit paired comparisons, provenance, original session logs and chart.
- `kaggle/notebook/stt-benchmark.ipynb`: documented analysis notebook, with optional inference stages.
- `docs/report_laptop.md`: original laptop report and cloud measurements.
- `docs/PROTOCOL.md`: original pre-registered protocol; the Kaggle extension and recovery are documented in `docs/LOG.md`.

The Kaggle notebook and its audio and code datasets stay private. No audio was republished.

Vendor context checked 6 October 2026: [Redux model card](https://huggingface.co/moondream/parakeet-redux) and [Photon local-runtime documentation](https://docs.moondream.ai/running-locally/). Their speed claims use other hardware and datasets; they are not interchangeable with these measurements.
