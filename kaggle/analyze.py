"""Generate the Kaggle supplement from scored CSVs; never edits raw measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'kaggle/results'
NAMES = {'redux': 'Parakeet Redux', 'tdt_nemo': 'Parakeet TDT v3 (NeMo)',
         'unified_nemo': 'Parakeet Unified (NeMo)', 'unified_q8': 'Parakeet Unified Q8 (GGUF)',
         'wturbo_fw': 'Whisper large-v3-turbo (CUDA)', 'wlv3_fw': 'Whisper large-v3 (CUDA)',
         'moon_hf': 'Moonshine base (CUDA)', 'tdt_q8': 'Parakeet TDT Q8 (GGUF)',
         'tdt_f16': 'Parakeet TDT F16 (GGUF)', 'moon_f16': 'Moonshine base F16 (GGUF)'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def transcripts(files):
    import json
    texts = {}
    for f in sorted(files, key=lambda p: p.stat().st_mtime):
        for line in f.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('kind') == 'clip' and r.get('pass', 0) == 0:
                texts[r['id']] = r['text']
    return texts


def identity(a, b):
    ids = a.keys() & b.keys()
    assert len(ids) == 147, 'Incomplete cross-machine comparison'
    return f'{sum(a[i] == b[i] for i in ids)} / {len(ids)}'


def main():
    m = pd.read_csv(OUT / 'metrics_accuracy.csv')
    s = pd.read_csv(OUT / 'metrics_speed.csv')
    p = pd.read_csv(OUT / 'per_clip.csv')
    p = p[~p.subset.str.startswith('D')]
    pairs = []
    for device in ('gpu', 'cpu'):
        a = p[(p.system == 'redux') & (p.device == device)].set_index('clip')
        for other, od in [('tdt_nemo', 'gpu'), ('unified_nemo', 'gpu'), ('wturbo_fw', 'gpu'), ('unified_q8', 'cpu')]:
            b = p[(p.system == other) & (p.device == od)].set_index('clip')
            assert set(a.index) == set(b.index) and len(a) == 78
            b = b.loc[a.index]
            assert np.array_equal(a.n.values, b.n.values)
            rng = np.random.default_rng(42)
            ea, eb, n = a.err.values, b.err.values, a.n.values
            idx = rng.integers(0, len(a), (10000, len(a)))
            diffs = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1) * 100
            pairs.append(dict(a='redux', device_a=device, b=other, device_b=od, clips=len(a),
                              diff_pts=100 * (ea.sum() - eb.sum()) / n.sum(),
                              ci_lo_pts=np.percentile(diffs, 2.5), ci_hi_pts=np.percentile(diffs, 97.5),
                              share_resamples_positive=float((diffs > 0).mean())))
    paired = pd.DataFrame(pairs)
    paired.to_csv(OUT / 'paired_kaggle.csv', index=False)
    acc = m[m.subset == 'ALL_PUBLIC'].sort_values('wer')
    acc_table = table(['System', 'Device', 'WER', '95% clip bootstrap CI'], [
        [NAMES[r.system], r.device, f'{100*r.wer:.2f}%', f'{100*r.ci_lo:.2f}–{100*r.ci_hi:.2f}%'] for r in acc.itertuples()])
    speed_table = table(['System', 'Device', 'RTFx', 'p50 / p95 (s)', 'Peak RAM (MiB)', 'GPU delta (MiB)'], [
        [NAMES[r.system], r.device, f'{r.rtfx:.2f}x', f'{r.latency_p50_s:.3f} / {r.latency_p95_s:.3f}',
         f'{r.peak_rss_mb:.0f}', f'{r.peak_vram_delta_mb:.0f}'] for r in s.sort_values(['device','rtfx'],ascending=[True,False]).itertuples()])
    comparison = table(['Comparison (Redux minus reference)', 'WER difference', '95% CI'], [
        [f'Redux {r.device_a} – {NAMES[r.b]} {r.device_b}', f'{r.diff_pts:+.2f} points',
         f'{r.ci_lo_pts:+.2f} to {r.ci_hi_pts:+.2f}'] for r in paired.itertuples()])
    sub = m[m.subset.isin(['A','B','B2','C','D','D_verbatim']) &
            ((m.system=='redux') | (m.system=='tdt_nemo'))].pivot(index=['system','device'],columns='subset',values='wer')
    subset_table = table(['System/device','YouTube A','Benchmark B','Accents B2','Noise C','Dictation script','Dictation consensus'], [
        [f'{NAMES[system]} {device}'] + [f'{100*row[k]:.2f}%' for k in ['A','B','B2','C','D','D_verbatim']]
        for (system,device),row in sub.iterrows()])
    main_pair = paired[(paired.device_a=='gpu') & (paired.b=='tdt_nemo')].iloc[0]
    redux_cpu = transcripts([OUT / 'raw/redux/cpu_accuracy.jsonl'])
    redux_gpu = transcripts([OUT / 'raw/redux/gpu_accuracy.jsonl'])
    redux_laptop = transcripts((ROOT / 'results/raw/redux').glob('*accuracy*.jsonl'))
    unified_laptop = transcripts((ROOT / 'results/raw/unified_q8').glob('*accuracy*.jsonl'))
    unified_kaggle = transcripts([OUT / 'raw/unified_q8/cpu_accuracy.jsonl'])
    identity_table = table(['Exact transcript-string comparison', 'Identical units'], [
        ['Redux laptop CPU versus Kaggle CPU', identity(redux_laptop, redux_cpu)],
        ['Redux Kaggle CPU versus GPU', identity(redux_cpu, redux_gpu)],
        ['Unified GGUF laptop versus Kaggle CPU', identity(unified_laptop, unified_kaggle)]])
    report = f'''# Parakeet Redux: Is a 178 MB Speech Model Useful in the Real World?

Abhash Chakraborty · 6 October 2026 · Final Kaggle supplement

**Verdict:** Redux is a practical option when a small model download and fast CPU transcription matter. It is also fast on the tested T4 GPU, but gives up some accuracy versus its original model. Its 178 MB weights do not imply a tiny runtime. The evidence supports a specific trade-off, not a universal winner.

## What was completed

The original Kaggle script produced 13 successful configurations. The separate repair script completed the 6 configurations affected by the PyAV and NeMo configuration bugs. The documented notebook then ended in `ERROR`, with no execution log or failure explanation exposed by Kaggle's API. No cause is inferred from that empty log.

The finalized measurements therefore combine **two completed inference sessions**, both on T4 hardware. They are not described as a clean single-session rerun. The recovered raw files are byte-identical to their archived/downloaded sources; `provenance.json` lists their SHA-256 hashes and source sessions. `validation.json` checks 8 accuracy configurations × 147 units, plus 11 speed configurations × 20 units × 3 passes, exact unit identities, durations, duplicate records, positive timings and completion records. Original failures remain in `sessions/`.

The final documented notebook **v5 completed successfully on Kaggle**. Its exported accuracy/speed scores agree with local scoring, all source provenance matches, and all eight explicit paired comparisons were saved. It defaults to validating and rescoring these saved measurements and performs no new inference in this mode. `RUN_BENCHMARK = True` opts into the original five inference stages; a GPU must also be enabled. That optional mode has not been newly executed in this completion pass.

## Hardware and method

The measurement host was an Intel Xeon at 2.00 GHz, 2 cores / 4 logical CPUs, 32 GB RAM and NVIDIA Tesla T4 (15,360 MiB), Ubuntu 24.04.1. Both sessions logged T4 and driver 580.178.04. Workers used 4 threads. The first session recorded AVX2 and AVX-512F but **no AVX-VNNI or AVX-512-VNNI**; Photon selected its AVX2 kernel. This is not a test of the advertised VNNI fast path. No Vulkan GPU driver was available, so GGUF references ran on the CPU.

All systems use the same 147 pre-segmented English audio units. Public accuracy covers 78 parent clips and 5,531 reference words: six YouTube clips, 50 benchmark utterances, 12 accent passages and 10 noisy derivatives. Four private dictated paragraphs are reported separately. Each speed configuration uses the same 20 benchmark clips, {s.subset_audio_s.iloc[0]:.2f} seconds of audio, a warm-up and three timed passes. RTFx is the median throughput; latency is per-clip p50/p95. RAM and GPU-memory deltas were sampled every 50 ms.

The same `scripts/score.py` normalizer and scorer as the laptop experiment were used. The supplemental paired comparisons below explicitly match system **and device**, with 10,000 resamples and seed 42. Different runtimes and precisions remain confounders: Photon vs NeMo is a practical system comparison and cannot isolate compression alone.

## Accuracy

{acc_table}

{subset_table}

Redux GPU scores **5.32% WER**, versus **4.72%** for the original Parakeet in NeMo. CPU Redux scores 5.50%. Device and runtime can change transcripts; the laptop's 10-clip device check does not establish universal CPU/GPU equivalence.

{comparison}

The central GPU comparison is **{main_pair.diff_pts:+.2f} WER points**, with 95% CI **{main_pair.ci_lo_pts:+.2f} to {main_pair.ci_hi_pts:+.2f}**. The interval includes zero, so this sample does not establish a difference at that confidence level. It also exceeds the pre-registered +0.5-point non-inferiority margin: near-equal accuracy is not demonstrated. The fraction of bootstrap resamples with a positive difference is not a posterior probability or a conventional p-value. Overlapping separate confidence intervals do not prove equivalence.

## Speed and resources

{speed_table}

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

{identity_table}

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
'''
    (ROOT / 'docs/report_kaggle.md').write_text(report, encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    for i, r in enumerate(acc.sort_values('wer', ascending=False).itertuples()):
        color = '#2563eb' if r.system == 'redux' else '#64748b'
        axes[0].errorbar(r.wer*100, i, xerr=[[100*(r.wer-r.ci_lo)],[100*(r.ci_hi-r.wer)]],fmt='o',color=color)
    axes[0].set_yticks(range(len(acc)),[NAMES[r.system]+' ['+r.device+']' for r in acc.sort_values('wer',ascending=False).itertuples()])
    axes[0].set_xlabel('WER %, with exploratory 95% clip bootstrap CI')
    axes[0].set_title('Accuracy: 78 public English clips')
    gpu = s[s.device=='gpu'].sort_values('rtfx')
    labels = [NAMES[x] for x in gpu.system]
    axes[1].barh(labels,gpu.rtfx,color=['#2563eb' if x=='redux' else '#64748b' for x in gpu.system])
    for i,v in enumerate(gpu.rtfx):axes[1].text(v+1,i,f'{v:.1f}x',va='center')
    axes[1].set_xlim(0,gpu.rtfx.max()*1.15); axes[1].set_xlabel('Audio seconds per processing second')
    axes[1].set_title('T4 throughput: 20 clips, median of 3 passes')
    fig.suptitle('Parakeet Redux: recovered original + repair measurements',fontweight='bold')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'summary.png',dpi=160);plt.close(fig)
    print('Generated docs/report_kaggle.md, paired_kaggle.csv and summary.png')


if __name__ == '__main__':main()
