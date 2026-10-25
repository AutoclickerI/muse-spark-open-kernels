"""Build a whole-layer leave-one-out check, using identical S14 weights/state for every arm.

Each reverted source is composed independently from S13 final. Full namespaces preserve decorator/global
bindings (notably nf_module's lru_cache); no hand-written inverse patch is needed. Current admitted patches
only add coded dense banks or change runtime kernels, so the prepared S14 state serves every arm exactly.
"""
import argparse
import ast
import io
import tokenize
import base64
import difflib
import hashlib
import importlib.util
import json
import lzma
import sys
import zlib
from pathlib import Path

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
sys.path.insert(0, str(R / 'tools'))
import h200_lab
from okc_run import rejected_words

spec = importlib.util.spec_from_file_location('s14_build', H / 'build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='s14ab')
    ap.add_argument('--without', nargs='*', default=[])
    ap.add_argument('--candidate', type=Path)
    args = ap.parse_args()
    src, meta = build.compose(args.without)
    candidate = args.candidate or D / 'cand_r3_best7_s14.py'
    assert candidate.read_text() == src, 'rebuild the candidate first'
    variants = {}
    for piece in build.pieces():
        if piece['name'] in args.without:continue
        text, _ = build.compose(args.without + [piece['name']])
        variants[piece['name']] = dict(source=text, active_tokens=piece['active_tokens'],
            sha256=hashlib.sha256(text.encode()).hexdigest())
    variants['S13'] = dict(source=build.BASE.read_text(), active_tokens=sorted({t for p in meta['patches'] for t in p['active_tokens']}),
                           sha256=build.BASE_SHA)
    # Reuse the reviewed S13/S10 timing helpers, excluding their obsolete lab and NT1 compiler.
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
    encoded = base64.b85encode(lzma.compress(bundle, preset=9)).decode()
    body = (helpers + '\nimport base64, json, lzma\n_S14_BUNDLE = json.loads(lzma.decompress(base64.b85decode('
            + repr(encoded) + ')))\nS14_HOST_SOURCE = _S14_BUNDLE["host"]\n'
            + (H / 'ab_lab.py.inc').read_text().replace('__S14_VARIANTS__', '_S14_BUNDLE["variants"]', 1))
    # Compress the complete helper/body as well as the source bundle, keeping request JSON below 1 MiB.
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    raw = base64.b85encode(lzma.compress(body.encode(), preset=9)).decode()
    bootstrap = ('import base64, lzma, hashlib, linecache\n'
                 '_s14_body = lzma.decompress(base64.b85decode(' + repr(raw) + ')).decode()\n'
                 'assert hashlib.sha256(_s14_body.encode()).hexdigest() == ' + repr(body_sha) + '\n'
                 '_s14_file = "<s14ab-body-' + body_sha[:12] + '>"\n'
                 'linecache.cache[_s14_file] = (len(_s14_body), None, _s14_body.splitlines(True), _s14_file)\n'
                 'exec(compile(_s14_body, _s14_file, "exec"), globals())\n'
                 '_s14_lab = lab\n'
                 'def lab(deadline, weights, state):\n    return _s14_lab(deadline, weights, state)\n')
    payload = H / (args.tag + '_payload.py')
    payload.write_text(bootstrap)
    embedded = [h200_lab.embed(payload)]
    # Only the outer lab host drops Python comments; the submitted candidate and embedded arms stay exact.
    # Assert identical Python AST, including every CUDA string, before using this smaller request envelope.
    outer_lines = src.splitlines(keepends=True)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and tok.start[0] != 1:
            line = tok.start[0] - 1
            outer_lines[line] = outer_lines[line][:tok.start[1]] + outer_lines[line][tok.end[1]:]
    outer = ''.join(outer_lines)
    assert ast.dump(ast.parse(outer)) == ast.dump(ast.parse(src))
    wrapped = outer.rstrip() + '\n\n' + h200_lab.WRAPPER.format(
        names=', '.join(f'{n} {sha}' for n, sha, _ in embedded), base='S14 ' + hashlib.sha256(src.encode()).hexdigest()[:8],
        payloads=embedded, budget=300 - h200_lab.MARGIN_S, limit=h200_lab.LIMIT)
    assert not rejected_words(wrapped), rejected_words(wrapped)
    compile(wrapped, '<S14 AB>', 'exec')
    out = D / 'lab' / ('lab_decode_' + args.tag + '.py')
    assert len(wrapped.encode()) <= 1_000_000, len(wrapped.encode())
    out.write_text(wrapped)
    manifest = dict(source=str(out.relative_to(R)), sha256=build.sha(out), host_sha256=build.sha(candidate),
        payload_sha256=build.sha(payload), source_bytes=len(wrapped.encode()), seeds=16, rounds=8, decode1_rounds=32, verify1_rounds=32,
        arms={k: {n: v for n, v in arm.items() if n != 'source'} for k, arm in variants.items()}, patches=meta['patches'])
    (H / (args.tag + '_manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest['source'], manifest['sha256'])


if __name__ == '__main__':
    main()
