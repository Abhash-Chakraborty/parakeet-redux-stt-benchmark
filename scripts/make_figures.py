"""Render the report's two charts as PNG (results/figures/) from the scored CSVs, for the Markdown report
and social posts. Grey data, one blue accent on Parakeet Redux."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
GREY, ACCENT, TEXT, MUTED, GRID = "#b9b8ae", "#2f6fd6", "#1f1f1f", "#5f5f5f", "#e6e6e6"
LABELS = {
    "el": "ElevenLabs Scribe v2 (cloud)", "oai": "OpenAI gpt-transcribe (cloud)",
    "unified_q8": "Parakeet Unified EN 0.6B", "wturbo_q8": "Whisper large-v3-turbo",
    "vercel": "Gemini 3.5 Transcribe (cloud)", "tdt_q8": "Parakeet TDT v3, 8-bit",
    "tdt_f16": "Parakeet TDT v3, original", "wlv3_q8": "Whisper large-v3",
    "redux": "Parakeet Redux (178 MB)", "moon_f16": "Moonshine base",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, zorder=0)


def wer_chart():
    m = pd.read_csv(ROOT / "results/metrics_accuracy.csv")
    m = m[m.subset == "ALL_PUBLIC"].sort_values("wer", ascending=False)
    ref = float(m[m.system == "tdt_f16"].wer.iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    for i, r in enumerate(m.itertuples()):
        c = ACCENT if r.system == "redux" else GREY
        ax.plot([r.ci_lo * 100, r.ci_hi * 100], [i, i], color=c, lw=2.2, zorder=2)
        ax.scatter(r.wer * 100, i, color=c, s=70, zorder=3)
        ax.text(r.ci_hi * 100 + 0.12, i, f"{r.wer * 100:.2f}%", va="center", color=TEXT if r.system == "redux" else MUTED,
                fontweight="bold" if r.system == "redux" else "normal")
    ax.axvline(ref, color="#8a8a8a", ls="--", lw=1, zorder=1)
    ax.text(ref + 0.05, -0.75, f"Original model: {ref:.2f}%", color=MUTED, fontsize=10)
    ax.set_ylim(-1.0, len(m) - 0.5)
    ax.set_yticks(range(len(m)), [LABELS[s] for s in m.system])
    for t, s in zip(ax.get_yticklabels(), m.system):
        t.set_color(TEXT if s == "redux" else MUTED)
        t.set_fontweight("bold" if s == "redux" else "normal")
    ax.set_xlim(2, 10)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Word error rate (normalized), lower is better", color=MUTED)
    style(ax)
    fig.suptitle("A 0.6B open model matches cloud accuracy; Redux trails its original by 0.8 points",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    ax.set_title("Normalized WER on 78 public clips (5,531 words). Dot = corpus WER, line = 95% bootstrap CI.",
                 loc="left", fontsize=10, color=MUTED, x=-0.35)
    fig.tight_layout()
    fig.savefig(OUT / "wer_overall.png", facecolor="white")


def rtfx_chart():
    s = pd.read_csv(ROOT / "results/metrics_speed.csv")
    keep = {"cpu_speed_t6": "CPU, 6 threads", "gpu_speed_t6": "GPU, GTX 1650 (Vulkan)"}
    s = s[s.config.isin(keep)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), dpi=200, sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, (cfg, title) in zip(axes, keep.items()):
        d = s[s.config == cfg].sort_values("rtfx")
        rows = list(d.itertuples())
        names = [LABELS[r.system].replace(" (178 MB)", "") for r in rows]
        if cfg == "gpu_speed_t6":
            names = ["Parakeet Redux"] + names
        for i, n in enumerate(names):
            r = next((x for x in rows if LABELS[x.system].startswith(n)), None)
            hot = n.startswith("Parakeet Redux")
            if r is None:
                ax.text(0.5, i, "not supported on this GPU", va="center", color=MUTED, style="italic")
                continue
            ax.barh(i, r.rtfx, color=ACCENT if hot else GREY, height=0.62, zorder=2)
            ax.text(r.rtfx + 0.8, i, f"{r.rtfx:.1f}x", va="center", color=TEXT if hot else MUTED,
                    fontweight="bold" if hot else "normal")
        ax.set_yticks(range(len(names)), names)
        for t, n in zip(ax.get_yticklabels(), names):
            t.set_color(TEXT if n.startswith("Parakeet Redux") else MUTED)
            t.set_fontweight("bold" if n.startswith("Parakeet Redux") else "normal")
        ax.axvline(1, color="#8a8a8a", ls="--", lw=1, zorder=1)
        ax.set_ylim(-0.8, len(names) - 0.4)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT)
        style(ax)
    axes[1].set_xlim(0, 72)
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0f}x")
    axes[1].set_xlabel("Real-time factor: audio seconds per processing second (higher is faster). Dashed line = real time.",
                       color=MUTED)
    fig.suptitle("On the GPU, Parakeet runs 2.5x faster than Redux does on the CPU",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.27, right=0.96, top=0.9, bottom=0.1)
    fig.savefig(OUT / "rtfx_by_device.png", facecolor="white")


def size_chart():
    """WER vs weights on disk for the local models, with the cloud models' WER range as a band."""
    m = pd.read_csv(ROOT / "results/metrics_accuracy.csv")
    m = m[m.subset == "ALL_PUBLIC"].set_index("system")
    sizes = {"moon_f16": 132, "redux": 178, "unified_q8": 731, "tdt_q8": 740, "wturbo_q8": 886,
             "tdt_f16": 1256, "wlv3_q8": 1669}
    offsets = {"moon_f16": (8, 6), "redux": (8, 6), "unified_q8": (-8, -16), "tdt_q8": (-10, 10),
               "wturbo_q8": (8, -14), "tdt_f16": (8, 4), "wlv3_q8": (-8, 8)}
    cloud = m.loc[["oai", "el", "vercel"], "wer"] * 100
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.axhspan(cloud.min(), cloud.max(), color="#d9e5f7", zorder=0)
    ax.text(105, cloud.max() + 0.08, f"Cloud APIs: {cloud.min():.2f} to {cloud.max():.2f}%", color=MUTED, fontsize=10)
    for sid, mb in sizes.items():
        w = m.loc[sid, "wer"] * 100
        hot = sid == "redux"
        ax.scatter(mb, w, s=90, color=ACCENT if hot else "#8f8e85", zorder=3)
        dx, dy = offsets[sid]
        ax.annotate(f"{LABELS[sid]}\n{w:.2f}%", (mb, w), textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left", fontsize=9.5, color=TEXT if hot else MUTED,
                    fontweight="bold" if hot else "normal")
    ax.set_xscale("log")
    ax.set_xlim(100, 2600)
    ax.set_ylim(3.0, 7.8)
    ax.set_xticks([100, 200, 500, 1000, 2000], ["100 MB", "200 MB", "500 MB", "1 GB", "2 GB"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Model weights on disk (log scale)", color=MUTED)
    ax.set_ylabel("Word error rate, 78 public clips (lower is better)", color=MUTED)
    style(ax)
    ax.grid(axis="y", color=GRID, zorder=0)
    fig.suptitle("Model size no longer predicts accuracy: two sub-1 GB models fall inside the cloud range", x=0.02, ha="left",
                 fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "wer_vs_size.png", facecolor="white")


if __name__ == "__main__":
    wer_chart()
    rtfx_chart()
    size_chart()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
