"""Recover completed Kaggle measurements, retaining source hashes and session provenance.

Never modifies laptop results or archived measurements. Does not run inference.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from validate_results import validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'kaggle/results'


def assemble():
    assert not OUT.exists(), 'Results exist: validate them instead of overwriting'
    sources = [('kaggle/results_run1', 'stt-benchmark-gpu v1', None),
               ('kaggle/_download_repair/out', 'stt-benchmark-gpu-v2 v1',
                {'unified_nemo', 'wturbo_fw', 'wlv3_fw'})]
    manifest = {'kind': 'recovered_measurements', 'date': '2026-10-06',
                'note': 'Original script session plus successful repair session; not a fresh single-session inference run.',
                'files': []}
    for folder, session, selected in sources:
        base = ROOT / folder
        for f in sorted(base.glob('raw/*/*.jsonl')):
            if selected is not None and f.parent.name not in selected:
                continue
            rel = f.relative_to(base)
            target = OUT / rel
            assert not target.exists(), target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            manifest['files'].append({'path': rel.as_posix(), 'source': folder,
                                      'session': session, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
        # Retain original failure statuses and logs in separate session directories.
        audit = OUT / 'sessions' / ('run1' if selected is None else 'repair')
        audit.mkdir(parents=True, exist_ok=True)
        for name in ('status.jsonl', 'main.log', 'env.json'):
            if (base / name).exists(): shutil.copy2(base / name, audit / name)
        if (base / 'logs').exists(): shutil.copytree(base / 'logs', audit / 'logs')
    shutil.copy2(ROOT / 'kaggle/results_run1/env.json', OUT / 'env.json')
    for name in ('data/units.csv', 'kaggle/bench.py', 'scripts/score.py', 'scripts/score_speed.py'):
        manifest.setdefault('local_input_hashes', {})[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (OUT / 'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    if not OUT.exists(): assemble()
    provenance = json.loads((OUT / 'provenance.json').read_text(encoding='utf-8'))
    hashes = {f['path']: f['sha256'] for f in provenance['files']}
    checked = validate(OUT, ROOT / 'data/units.csv', hashes)
    (OUT / 'validation.json').write_text(json.dumps(checked, indent=2), encoding='utf-8')
    env = dict(os.environ, STT_RESULTS='kaggle/results', MPLBACKEND='Agg')
    for script in ('score.py', 'score_speed.py'):
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script)], env=env, check=True)
    print('Validated 8 accuracy and 11 speed configurations; all source hashes match.')


if __name__ == '__main__': main()
