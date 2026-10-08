"""Score every accuracy run in results/raw (PROTOCOL.md section 5).

- Hypotheses of a clip's chunks are joined in chunk order, then compared with the clip reference.
- Normalized WER/CER: Whisper EnglishTextNormalizer on both sides (Open ASR Leaderboard convention).
- Formatted WER: lowercase, punctuation .,?!;: kept as separate tokens; only on subsets whose references are
  punctuated (A, B2, D).
- Corpus WER = total errors / total reference words; 95% CI by bootstrap over clips (10,000, seed 42).
- Paired bootstrap of WER differences between systems on the same clips.

Outputs: results/per_clip.csv, results/metrics_accuracy.csv, results/paired.csv
"""
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[1]
RES = os.environ.get("STT_RESULTS", "results")  # results root; Kaggle outputs use kaggle/_out
NORM = EnglishTextNormalizer()
FORMATTED_SUBSETS = {"A", "B2", "D"}
N_BOOT, SEED = 10_000, 42
PAIRS = [("redux", "tdt_f16"), ("redux", "tdt_q8"), ("tdt_q8", "tdt_f16"), ("redux", "unified_q8"),
         ("unified_q8", "tdt_f16"), ("redux", "wlv3_q8"), ("unified_q8", "wlv3_q8")]


def d_prenorm(text):
    """Subset D only: formatting variants that are not recognition errors. Times written 3.30 vs 3:30, and
    '11 labs' / 'eleven labs' vs 'ElevenLabs'. Applied identically to reference and hypothesis."""
    text = re.sub(r"\b(\d{1,2})[.:](\d{2})\b", r"\1:\2", text)
    return re.sub(r"\b(11|eleven)\s*labs\b", "ElevenLabs", text, flags=re.I)


def fmt_tokens(text):
    text = text.lower().replace("’", "'")
    return " ".join(re.findall(r"[a-z0-9']+|[.,?!;:]", text))


def counts(ref, hyp):
    if not ref.strip():
        return None
    if not hyp.strip():
        n = len(ref.split())
        return dict(hits=0, sub=0, dele=n, ins=0, n=n)
    o = jiwer.process_words(ref, hyp)
    return dict(hits=o.hits, sub=o.substitutions, dele=o.deletions, ins=o.insertions,
                n=o.hits + o.substitutions + o.deletions)


def cer_counts(ref, hyp):
    if not hyp.strip():
        return len(ref), len(ref)
    o = jiwer.process_characters(ref, hyp)
    return o.substitutions + o.deletions + o.insertions, o.hits + o.substitutions + o.deletions


def load_refs():
    """Clip references; subset D also gets a second, verbatim reference (what the speaker actually said,
    from data/own_voice/verbatim.txt) scored as subset 'D_verbatim'."""
    refs = {}
    for m in ["manifest_youtube.csv", "manifest_bench.csv", "manifest_accent.csv", "manifest_own.csv"]:
        p = ROOT / "data" / m
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                refs[r["id"]] = r
    vpath = ROOT / "data/own_voice/verbatim.txt"
    if vpath.exists():
        lines = [l.strip() for l in vpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, line in enumerate(lines):
            pid = f"D_own_{i:02d}"
            if pid in refs:
                refs[pid + "@verbatim"] = {**refs[pid], "subset": "D_verbatim", "reference": line}
    return refs


def load_hyps():
    """{(system, device): {unit_id: text}} from every *accuracy*.jsonl (pass 0).

    Files are read oldest first, so a rerun of a unit overrides the earlier output (deviation D2: subset A was
    rerun after peak normalization; the superseded outputs stay on disk for audit)."""
    runs = defaultdict(dict)
    for f in sorted((ROOT / RES / "raw").glob("*/*accuracy*.jsonl"), key=lambda p: p.stat().st_mtime):
        system, config = f.parent.name, f.stem
        device = config.split("_")[0]
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if d.get("kind") == "clip" and d.get("pass", 0) == 0:
                runs[(system, device)][d["id"]] = d["text"]
    return runs


def main():
    refs, runs = load_refs(), load_hyps()
    units = pd.read_csv(ROOT / "data/units.csv")
    parent_units = units.sort_values("unit_id").groupby("parent_id").unit_id.apply(list).to_dict()
    rows = []
    for (system, device), hyps in runs.items():
        for ref_id, r in refs.items():
            pid = ref_id.split("@")[0]
            uids = parent_units.get(pid, [])
            if not uids or not all(u in hyps for u in uids):
                continue  # clip not (fully) run by this system
            hyp = " ".join(hyps[u].strip() for u in uids)
            ref_text = r["reference"]
            if r["subset"].startswith("D"):
                ref_text, hyp = d_prenorm(ref_text), d_prenorm(hyp)
            ref_n, hyp_n = NORM(ref_text), NORM(hyp)
            c = counts(ref_n, hyp_n)
            ce, cn = cer_counts(ref_n, hyp_n)
            row = dict(system=system, device=device, clip=ref_id, subset=r["subset"], group=r["group"],
                       audio_s=float(r["duration_s"]), **c, char_err=ce, char_n=cn, hyp=hyp, ref=ref_text)
            if r["subset"] in FORMATTED_SUBSETS or r["subset"] == "D_verbatim":
                fc = counts(fmt_tokens(ref_text), fmt_tokens(hyp))
                row.update(fmt_err=fc["sub"] + fc["dele"] + fc["ins"], fmt_n=fc["n"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["err"] = df["sub"] + df["dele"] + df["ins"]
    df.drop(columns=["hyp", "ref"]).to_csv(ROOT / RES / "per_clip.csv", index=False)
    df[["system", "device", "clip", "hyp", "ref"]].to_csv(ROOT / RES / "per_clip_text.csv", index=False)

    rng = np.random.default_rng(SEED)
    out = []
    for (system, device), g in df.groupby(["system", "device"]):
        for subset, h in [("ALL_PUBLIC", g[~g.subset.str.startswith("D")])] + list(g.groupby("subset")):
            if h.empty:
                continue
            e, n = h["err"].to_numpy(), h["n"].to_numpy()
            idx = rng.integers(0, len(h), size=(N_BOOT, len(h)))
            boot = e[idx].sum(1) / n[idx].sum(1)
            fm = h.dropna(subset=["fmt_n"]) if "fmt_n" in h else h.iloc[0:0]
            out.append(dict(system=system, device=device, subset=subset, clips=len(h), ref_words=int(n.sum()),
                            wer=e.sum() / n.sum(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                            cer=h["char_err"].sum() / h["char_n"].sum(),
                            sub=h["sub"].sum() / n.sum(), dele=h["dele"].sum() / n.sum(), ins=h["ins"].sum() / n.sum(),
                            fmt_wer=(fm["fmt_err"].sum() / fm["fmt_n"].sum()) if len(fm) else np.nan))
    m = pd.DataFrame(out).sort_values(["subset", "wer"])
    m.to_csv(ROOT / RES / "metrics_accuracy.csv", index=False)

    # paired bootstrap: WER(a) - WER(b) over the clips both systems ran (best device per system: gpu, else cpu)
    best = df.sort_values("device", ascending=False).drop_duplicates(["system", "clip"])  # 'gpu' > 'cpu'
    paired = []
    for a_sys, b_sys in PAIRS:
        a, b = best[best.system == a_sys], best[best.system == b_sys]
        j = a.merge(b, on="clip", suffixes=("_a", "_b"))
        j = j[~j.subset_a.str.startswith("D")]
        if j.empty:
            continue
        ea, eb, n = j["err_a"].to_numpy(), j["err_b"].to_numpy(), j["n_a"].to_numpy()
        idx = rng.integers(0, len(j), size=(N_BOOT, len(j)))
        diff = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
        paired.append(dict(a=a_sys, b=b_sys, clips=len(j), wer_a=ea.sum() / n.sum(), wer_b=eb.sum() / n.sum(),
                           diff_pts=100 * (ea.sum() - eb.sum()) / n.sum(), ci_lo_pts=100 * np.percentile(diff, 2.5),
                           ci_hi_pts=100 * np.percentile(diff, 97.5), p_a_worse=float((diff > 0).mean())))
    pd.DataFrame(paired).to_csv(ROOT / RES / "paired.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(m[m.subset == "ALL_PUBLIC"].round(4).to_string(index=False))
        print(pd.DataFrame(paired).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
