"""Build a whole-layer leave-one-out check, using identical S12 weights/state for every arm.

Each reverted source is composed independently from S11 final. Full namespaces preserve decorator/global
bindings (notably nf_module's lru_cache); no hand-written inverse patch is needed. Current admitted patches
only add coded dense banks or change runtime kernels, so the prepared S12 state serves every arm exactly.
"""
import argparse
import base64
import difflib
import hashlib
import importlib.util
import json
import sys
import zlib
from pathlib import Path

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
sys.path.insert(0, str(R / 'tools'))
import h200_lab
from okc_run import rejected_words

spec = importlib.util.spec_from_file_location('s12_build', H / 'build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='s12ab')
    args = ap.parse_args()
    src, meta = build.compose()
    candidate = D / 'cand_r3_best7_s12.py'
    assert candidate.read_text() == src, 'rebuild the candidate first'
    variants = {}
    for piece in build.pieces():
        text, _ = build.compose([piece['name']])
        variants[piece['name']] = dict(source=text, active_tokens=piece['active_tokens'],
            sha256=hashlib.sha256(text.encode()).hexdigest())
    variants['S11'] = dict(source=build.BASE.read_text(), active_tokens=sorted({t for p in build.pieces() for t in p['active_tokens']}),
                           sha256=build.BASE_SHA)
    # Reuse the reviewed S11/S10 timing helpers, excluding their obsolete lab and NT1 compiler.
    helpers = (D / 'dev/s10/ablate_lab.py.inc').read_text().split('\ndef lab(deadline, weights, state):', 1)[0]
    helpers = helpers.replace('__NT1_CUDA__', "''", 1).replace('__NT1_CPP__', "''", 1)
    # One shared source plus lossless line edits: complete source copies exceed the server request limit.
    host_lines = src.splitlines(keepends=True)
    packed = {}
    for name, arm in variants.items():
        lines = arm['source'].splitlines(keepends=True)
        edits = [(a, b, ''.join(lines[c:d])) for op, a, b, c, d in
                 difflib.SequenceMatcher(None, host_lines, lines, autojunk=False).get_opcodes() if op != 'equal']
        restored = list(host_lines)
        for a, b, text in reversed(edits):restored[a:b] = [text]
        assert ''.join(restored) == arm['source'], name
        packed[name] = {k: v for k, v in arm.items() if k != 'source'} | dict(edits=edits)
    bundle = json.dumps(dict(host=src, variants=packed), separators=(',', ':')).encode()
    encoded = base64.b85encode(zlib.compress(bundle, 9)).decode()
    body = (helpers + '\nimport base64, json, zlib\n_S12_BUNDLE = json.loads(zlib.decompress(base64.b85decode('
            + repr(encoded) + ')))\nS12_HOST_SOURCE = _S12_BUNDLE["host"]\n'
            + (H / 's12ab_lab.py.inc').read_text().replace('__S12_VARIANTS__', '_S12_BUNDLE["variants"]', 1))
    payload = H / (args.tag + '_payload.py')
    payload.write_text(body)
    embedded = [h200_lab.embed(payload)]
    wrapped = src.rstrip() + '\n\n' + h200_lab.WRAPPER.format(
        names=', '.join(f'{n} {sha}' for n, sha, _ in embedded), base='S12 ' + hashlib.sha256(src.encode()).hexdigest()[:8],
        payloads=embedded, budget=300 - h200_lab.MARGIN_S, limit=h200_lab.LIMIT)
    assert not rejected_words(wrapped), rejected_words(wrapped)
    compile(wrapped, '<S12 AB>', 'exec')
    out = D / 'lab' / ('lab_decode_' + args.tag + '.py')
    out.write_text(wrapped)
    manifest = dict(source=str(out.relative_to(R)), sha256=build.sha(out), host_sha256=build.sha(candidate),
        payload_sha256=build.sha(payload), seeds=16, rounds=8,
        arms={k: {n: v for n, v in arm.items() if n != 'source'} for k, arm in variants.items()}, patches=meta['patches'])
    (H / (args.tag + '_manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest['source'], manifest['sha256'])


if __name__ == '__main__':
    main()
