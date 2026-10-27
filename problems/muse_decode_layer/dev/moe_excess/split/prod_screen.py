"""Rehearse the exact packed 64x16 production batch locally with S22's normal glue (sm_120)."""
import builtins
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
import prod_lab
build = prod_lab.build

H, R, P = build.H, build.R, build.P


def main():
    forced = '--forced' in sys.argv
    mode = 'forced' if forced else 'normal'
    screen = build.load(P / 'dev/s22/screen.py', 'p3b_split_screen')
    codec = build.load(P / 'dev/s21/evidence.py', 'p3b_night_codec')
    manifest = json.loads((H / 'prod_manifest.json').read_text())
    path = H / 'out/prod_wrapper.py'
    source = path.read_text()
    assert build.sha(source) == manifest['wrapper_sha256']
    module = types.ModuleType('submission')
    module.__file__ = str(path)
    emitted = []
    calls = []
    def emit(*args, **kwargs):
        emitted.append(kwargs.get('sep', ' ').join(map(str, args)) + kwargs.get('end', '\n'))
        return builtins.print(*args, **kwargs)
    module.print = emit
    def install():
        def local(m):
            record = screen.install(m, forced)
            calls.append(record)
        module.S16_LOCAL_INSTALL = local
        module.S21_LOCAL_ARCHIVE = lambda raw: (H / 'out' / ('prod_' + mode + '_raw.json')).write_text(json.dumps(raw))
        original = module._lab_run_one
        def run(name, src, deadline, weights, state):
            try:
                return original(name, src, time.monotonic() + 3600, weights, state)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
        module._lab_run_one = run
    module._P3B_LOCAL = install
    marker = '\n_lab_import_start = '
    assert source.count(marker) == 1
    start = time.monotonic()
    exec(compile(source.replace(marker, '\n_P3B_LOCAL()' + marker, 1), str(path), 'exec'), module.__dict__)
    try:
        module.prepare(None)
        raise AssertionError('lab-only prepare returned')
    except RuntimeError as exc:
        assert 'lab-only host' in str(exc), exc
    sys.path.insert(0, str(R / 'tools'))
    from lab_packet import paired_evidence, read_packets
    from promotion import _protocol_arm_valid
    # Same exact public tail boundary as P4, with room for a trailing error/log.
    tail = (''.join(emitted) + 'P' * 4000)[-16000:]
    packets = read_packets(tail)
    assert len(packets) == 1 and packets[0]['source_sha256'] == manifest['payload_sha256'][:16]
    evidence = paired_evidence(tail, 'P3B_SPLIT')
    assert evidence is not None, tail[-2000:]
    raw = codec.s21_decode_samples(codec.s21_unpack_evidence(evidence))
    assert all(_protocol_arm_valid(raw, manifest, name, 64) for name in ('A', 'A2', *manifest['arms']))
    assert raw['complete'] and not raw['failures'], raw['failures']
    assert raw['task_checks'] == manifest['expected_TASK_checks']
    assert raw['nearby_task_checks'] == manifest['nearby_expected_checks']
    result = dict(passed=True, mode=mode, wrapper_sha256=manifest['wrapper_sha256'],
                  payload_sha256=manifest['payload_sha256'], TASK=raw['task_checks'],
                  nearby=raw['nearby_task_checks'], protocol=raw['measurement_protocol'],
                  exact_tail_transport=True, wire_bytes=max(len(s.encode()) for s in emitted if s.startswith('LAB_PACKET ')),
                  elapsed_s=round(time.monotonic() - start, 2),
                  limitation='Local sm120/stand-ins; no native performance claim')
    (H / ('prod_' + mode + '_validation.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
