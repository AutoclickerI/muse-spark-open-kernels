"""Asserted S15 composition on exact S14 final; no runtime fallback or source checkout."""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]
D = H.parents[1]
BASE = D / 'cand_r3_best7_s14_final.py'
BASE_SHA = 'f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
S11 = D / 'cand_r3_best7_s11_final.py'
S11_SHA = 'acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7'
PIECES = {'proj': [1, 8, 16, 32], 'last': [1], 'regmax': [128, 256]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new, 1)


def assignments(source):
    return {t.id: n for n in ast.parse(source).body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}


def replace_assignment(source, name, expected, value):
    node = assignments(source)[name]
    assert ast.literal_eval(node.value) == expected, name
    lines = source.splitlines(True)
    lines[node.lineno-1:node.end_lineno] = [name + ' = ' + repr(value) + '\n']
    return ''.join(lines)


def apply_unified(source, patch):
    """Apply the frozen unified patch to its original source, asserting every old line."""
    original = source.splitlines(True)
    lines = patch.splitlines(True)
    out, cursor, i = [], 0, 2
    while i < len(lines):
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[i])
        assert match, lines[i]
        start = int(match[1])-1
        out.extend(original[cursor:start]); cursor = start; i += 1
        old_count = new_count = 0
        while i < len(lines) and not lines[i].startswith('@@ '):
            line = lines[i]; i += 1
            if line.startswith('\\ No newline'):continue
            assert line[0] in ' +-'
            if line[0] in ' -':
                assert original[cursor] == line[1:], (cursor, original[cursor][:100], line[:100])
                cursor += 1; old_count += 1
            if line[0] in ' +':out.append(line[1:]); new_count += 1
        assert old_count == int(match[2] or 1) and new_count == int(match[4] or 1)
    out.extend(original[cursor:])
    return ''.join(out)


def patch_last(source):
    old = S11.read_text()
    changed = apply_unified(old, (H / 'last_decode1.patch').read_text())
    tail = changed[changed.index('\n\nimport hashlib\ncs_CUDA='):]
    source = once(source, '        native_pool.submit(_vq8_module)  # vq8 rows (q_len 8)\n',
                  '        native_pool.submit(_vq8_module)  # vq8 rows (q_len 8)\n        native_pool.submit(cs_module)\n')
    source = once(source, "    state['vq8_rows']=_vq8_prepare(device)\n",
                  "    state['vq8_rows']=_vq8_prepare(device)\n    state['cluster_rows']=_cluster_prepare(device)\n")
    anchor = "def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n"
    source = once(source, anchor, anchor + "    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state.get('cluster_rows',()):\n        return _cluster_attention(qkv,data)\n")
    source = replace_assignment(source, '_PP_ROWS', ((1, 1, False), (1, 1, True)), ((1, 1, False),))
    return source + tail


def patch_regmax(source):
    original = S11.read_text()
    changed = apply_unified(original, (H / 'regmax_full_verify.patch').read_text())
    before, after = assignments(original), assignments(changed)
    for name in ('VQ8_CUDA', '_VQ8_ROWS'):
        source = replace_assignment(source, name, ast.literal_eval(before[name].value), ast.literal_eval(after[name].value))
    source = once(source, "    state['vq8_rows']=_vq8_prepare(device)\n",
                  "    state['vq8_rows']=_vq8_prepare(device)\n    assert state['vq8_rows']==_VQ8_ROWS, 'regmax attention admission failed'\n")
    return once(source, 'muse_vq8_c262e4c5f9c3_{arch}', 'muse_vq8_9a23ca9a98dd_{arch}')


def compose(without=(), without_role=None):
    assert sha(BASE) == BASE_SHA
    assert sha(S11)==S11_SHA
    for path, expected in json.loads((H / 'dependencies.json').read_text()).items():
        assert sha(H/path) == expected, path
    assert set(without) <= PIECES.keys()
    assert without_role in (None, 'qkv', 'o', 'unpooler')
    source = BASE.read_text()
    meta = dict(base=str(BASE.relative_to(R)), base_sha256=BASE_SHA, patches=[])
    for name, patch in (('last', patch_last), ('regmax', patch_regmax)):
        if name not in without:
            source = patch(source)
            meta['patches'].append(dict(name=name, active_tokens=PIECES[name]))
    if 'proj' not in without:
        selection = [tuple(row) for row in json.loads((H/'proj/selection.json').read_text())['selected']]
        selection = [key for key in selection if not (without_role == 'qkv' and key[1] in ('qkv3','qkv_gate')
                     or without_role in ('o','unpooler') and key[1] == without_role)]
        spec = importlib.util.spec_from_file_location('s15_nf_patch', H/'proj/patch.py')
        patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
        projected = patch.patch(BASE.read_text(), selection)
        assert projected.startswith(BASE.read_text())
        source += projected[len(BASE.read_text()):]
        assert 'Permission is hereby granted, free of charge' in source
        assert 'Nathen Fernandes' in source
        meta['patches'].append(dict(name='proj', active_tokens=sorted({t for t, role in selection}), selected=selection))
    compile(source, '<S15>', 'exec')
    meta.update(sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()))
    return source, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--without', nargs='*', default=[])
    ap.add_argument('--without-role', choices=('qkv','o','unpooler'))
    ap.add_argument('--out', type=Path, default=D/'cand_r3_best7_s15.py')
    ap.add_argument('--manifest', type=Path, default=H/'manifest.json')
    args = ap.parse_args()
    source, meta = compose(args.without, args.without_role)
    import sys
    sys.path.insert(0, str(R/'tools'))
    from okc_run import rejected_words
    assert not rejected_words(source), rejected_words(source)
    assert meta['bytes'] < 1_000_000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    meta['source'] = str(args.out.resolve().relative_to(R))
    args.manifest.write_text(json.dumps(meta, indent=2)+'\n')
    print(meta['source'], meta['sha256'], meta['bytes'])


if __name__ == '__main__':
    main()
