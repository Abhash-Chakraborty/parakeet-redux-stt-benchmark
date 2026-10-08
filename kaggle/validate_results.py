"""Fail closed on missing, duplicate or truncated benchmark measurements."""
import csv
import hashlib
import json
import math
from pathlib import Path

ACCURACY = [('redux', 'cpu'), ('redux', 'gpu'), ('tdt_nemo', 'gpu'),
            ('unified_nemo', 'gpu'), ('wturbo_fw', 'gpu'), ('wlv3_fw', 'gpu'),
            ('moon_hf', 'gpu'), ('unified_q8', 'cpu')]
SPEED = [('redux', 'cpu'), ('redux', 'gpu'), ('tdt_nemo', 'gpu'),
         ('unified_nemo', 'gpu'), ('wturbo_fw', 'gpu'), ('wlv3_fw', 'gpu'),
         ('moon_hf', 'gpu'), ('tdt_q8', 'cpu'), ('tdt_f16', 'cpu'),
         ('unified_q8', 'cpu'), ('moon_f16', 'cpu')]


def validate(results, units_file, expected_hashes=None):
    results = Path(results)
    units = list(csv.DictReader(open(units_file, encoding='utf-8')))
    ids = {u['unit_id'] for u in units}
    assert len(ids) == len(units) == 147, 'Unexpected input manifest'
    b = [u for u in units if u['subset'] == 'B']
    speed = {u['unit_id'] for g in sorted({u['group'] for u in b})
             for u in [x for x in b if x['group'] == g][:4]}
    assert len(speed) == 20
    durations = {u['unit_id']: float(u['duration_s']) for u in units}
    files = []
    for which, configs in [('accuracy', ACCURACY), ('speed', SPEED)]:
        for system, device in configs:
            matches = list((results / 'raw' / system).glob(f'{device}_{which}*.jsonl'))
            assert len(matches) == 1, (system, device, which, 'ambiguous/missing file')
            f = matches[0]
            records = [json.loads(line) for line in f.read_text(encoding='utf-8').splitlines()]
            assert records[0]['kind'] == 'load' and records[-1]['kind'] == 'self_resources', f
            load = records[0]
            assert (load['system'], load['device']) == (system, device), f
            assert load['threads'] == 4, (f, 'Expected 4-thread Kaggle run')
            clips = [r for r in records if r['kind'] == 'clip']
            want = ids if which == 'accuracy' else speed
            passes = range(1 if which == 'accuracy' else 3)
            assert len(clips) == len(want) * len(passes), f
            assert {r['pass'] for r in clips} == set(passes), f
            for k in passes:
                got = [r['id'] for r in clips if r['pass'] == k]
                assert len(got) == len(set(got)) and set(got) == want, (f, k)
            for r in clips:
                assert isinstance(r['text'], str), f
                assert math.isfinite(r['wall_s']) and r['wall_s'] > 0, f
                assert abs(r['audio_s'] - durations[r['id']]) < 1e-6, f
            rel = f.relative_to(results).as_posix()
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            if expected_hashes is not None:
                assert expected_hashes[rel] == sha, (f, 'SHA-256 mismatch')
            files.append({'path': rel, 'sha256': sha, 'clips': len(clips)})
    assert len(list((results / 'raw').glob('*/*.jsonl'))) == len(files), 'Unexpected raw file'
    return {'ok': True, 'accuracy_configurations': len(ACCURACY),
            'speed_configurations': len(SPEED), 'accuracy_units_each': len(ids),
            'speed_units_each': len(speed), 'speed_passes': 3, 'files': files}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('results'); p.add_argument('units')
    a = p.parse_args()
    print(json.dumps(validate(a.results, a.units), indent=2))
