"""Build the public copy of this repo in another directory.

Copies every tracked or unignored file except working notes, and replaces the subset A (YouTube) caption
and transcript text with a placeholder. Scores, URLs and timestamps are kept. Then checks that no 40-character
window of any subset A reference survives anywhere in the copy.

    envs/cloud/Scripts/python.exe scripts/export_public.py <out_dir>

<out_dir> may be a clone of the public repo: everything in it except .git is replaced.
"""
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = {"HANDOFF.md", "docs/blog_agent_brief.md", "docs/source_summary_clark.md"}
WITHHELD = "[withheld: YouTube captions]"


def is_a(value):
    return isinstance(value, str) and value.startswith("A_")


def strip_csv(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return text
    fields = list(rows[0].keys())
    key = next((k for k in ("id", "clip", "unit_id", "parent_id") if k in fields), None)
    text_cols = [c for c in ("reference", "hyp", "ref", "text") if c in fields]
    if key is None or not text_cols:
        return text
    for r in rows:
        if is_a(r[key]):
            for c in text_cols:
                r[c] = WITHHELD
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def strip_jsonl(text):
    lines = []
    for line in text.splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("kind") == "clip" and is_a(d.get("id")):
                d["text"] = WITHHELD
            if d.get("kind") == "load" and is_a(d.get("first_clip")):
                d["first_text"] = WITHHELD
            line = json.dumps(d)
        lines.append(line)
    return "\n".join(lines) + "\n"


def a_windows():
    refs = [r["reference"] for r in csv.DictReader(open(ROOT / "data/manifest_youtube.csv", encoding="utf-8"))]
    return {ref[i:i + 40] for ref in refs for i in range(0, max(len(ref) - 40, 1), 25)}


def main(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    for f in files:
        src = ROOT / f
        if f in EXCLUDE or not src.exists():
            continue
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.endswith(".csv"):
            dst.write_text(strip_csv(src.read_text(encoding="utf-8")), encoding="utf-8")
        elif f.endswith(".jsonl") or f.endswith(".jsonl.txt"):
            dst.write_text(strip_jsonl(src.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(src, dst)

    windows = a_windows()
    leaks = []
    for p in out.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            try:
                t = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(w in t for w in windows):
                leaks.append(p.relative_to(out).as_posix())
    if leaks:
        sys.exit("subset A text still present in:\n  " + "\n  ".join(leaks))
    print(f"exported {len(files) - len(EXCLUDE)} files to {out}; no subset A text found")


if __name__ == "__main__":
    main(sys.argv[1])
