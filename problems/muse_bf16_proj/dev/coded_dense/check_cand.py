"""Normal / forced local checks of a decode candidate carrying the coded patch: the integration lane's
problems/muse_decode_layer/dev/s9/audit.py (local_eval tests + loads; --forced forces the Hopper glue through
best7/force.py), plus a count of coded GEMV calls by (tokens, N, S), eager and captured, to show which rows run coded.

    CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/coded_dense/check_cand.py <cand.py> <out_dir> [--forced]
"""
import importlib.util
import hashlib
import ast
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R / 'tools'))
import local_eval  # noqa: E402


def native_units(path):
    """The source/flag identity of exactly the literal CUDA units local_eval compiles."""
    tree = ast.parse(path.read_text())
    templated, flags = set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in ('format', 'replace', 'format_map') and isinstance(n.value, ast.Name):
            templated.add(n.value.id)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and isinstance(n.left, ast.Name):
            templated.add(n.left.id)
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', getattr(n.func, 'id', '')) == 'load_inline':
            kw = {k.arg: k.value for k in n.keywords}; src = kw.get('cuda_sources'); extra = []
            for key, prefix in (('extra_cuda_cflags', ''), ('extra_include_paths', '-I')):
                try:extra += [prefix + str(v) for v in ast.literal_eval(kw[key])]
                except (KeyError, ValueError):pass
            for node in src.elts if isinstance(src, (ast.List, ast.Tuple)) else [src]:
                if isinstance(node, ast.Name):flags[node.id] = extra
    return {n.targets[0].id: (n.value.value, flags.get(n.targets[0].id, [])) for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            and '#include' in n.value.value and n.targets[0].id not in templated
            and (n.targets[0].id in flags or '__global__' in n.value.value or 'torch/extension.h' in n.value.value)}


def main():
    cand, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    forced = '--forced' in sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    # Native compilation is independent of the forced Python dispatch. Reuse the completed
    # normal compile proof only for the byte-identical candidate; all runtime checks still run.
    normal = out.parent / 'normal'
    if forced and (normal / 'audit.json').exists():
        proof = json.loads((normal / 'audit.json').read_text())
        if proof['sha256'] == hashlib.sha256(cand.read_bytes()).hexdigest():
            checks = json.loads((normal / 'normal_local.json').read_text())['static']
            assert all(c['ok'] for c in checks), checks
            native = [c for c in checks if c['name'] not in ('undefined_names', 'call_signatures')]
            local_eval.hopper_compile = lambda path: native
            print('NATIVE_COMPILE_PROOF_REUSED', proof['sha256'], len(native), flush=True)
    # Additive direct-code rows leave every literal CUDA unit unchanged. The new dynamic
    # decoder has its own sm90 proof/H200 receipt; reuse only identical completed units here.
    base = Path(__file__).parent / 's13_coded_local.py'
    proof_dir = Path(__file__).parent / 's13_check/normal'
    if base.exists() and (proof_dir / 'audit.json').exists():
        proof = json.loads((proof_dir / 'audit.json').read_text())
        if proof['sha256'] == hashlib.sha256(base.read_bytes()).hexdigest() and native_units(cand) == native_units(base):
            checks = json.loads((proof_dir / 'normal_local.json').read_text())['static']
            assert all(c['ok'] for c in checks)
            native = [c for c in checks if c['name'] not in ('undefined_names', 'call_signatures')]
            assert set(native_units(cand)) == {c['name'] for c in native}
            local_eval.hopper_compile = lambda path: native
            print('NATIVE_UNITS_PROOF_REUSED', proof['sha256'], len(native), flush=True)
    spec = importlib.util.spec_from_file_location('s9_audit', R / 'problems/muse_decode_layer/dev/s9/audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    counts = {}
    codes = {}
    original = local_eval.load_module

    def loader(name, path):
        m = original(name, path)
        if name == 'submission':
            def counter(gemv, family):
                def counted(x, c, S, **kw):
                    row = counts.setdefault(f"{family}:T{x.shape[0]}:N{c['N']}:K{c['K']}:S{S}", dict(eager=0, captured=0))
                    row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager'] += 1
                    return gemv(x, c, S, **kw)
                return counted
            m.cd_gemv = counter(m.cd_gemv, 'cp')
            if hasattr(m, 'wg_gemv'):m.wg_gemv = counter(m.wg_gemv, 'wg')
            if hasattr(m, 'dcp_gemv'):m.dcp_gemv = counter(m.dcp_gemv, 'direct')
            prepare = m.prepare

            def checked_prepare(weights):
                state = prepare(weights)
                # Exercise the judge's tensor-clone protocol, including new weight addresses.
                state = dict(state)
                state['coded_dense'] = {
                    role: {k: v.clone() if isinstance(v, m.torch.Tensor) else v for k, v in c.items()}
                    for role, c in state['coded_dense'].items()}
                state['direct_coded'] = {role: {k: v.clone() if isinstance(v, m.torch.Tensor) else v for k, v in c.items()}
                                         for role, c in state.get('direct_coded', {}).items()}
                banks = list(state['coded_dense'].items()) + [(('direct', role), c) for role, c in state['direct_coded'].items()]
                for key, c in banks:
                    family, role = (key[0], key[1]) if isinstance(key, tuple) else ('cp', key)
                    state[role] = state[role].clone()
                    decode = m.dcp_decode if family == 'direct' else m.wg_decode if family == 'wg' else m.cd_decode
                    assert m.torch.equal(decode(c).view(m.torch.int16), state[role].view(m.torch.int16)), key
                    expected = dict(coded=m.torch.uint8, scales=m.torch.float32, eoff=m.torch.int32,
                                    esc=m.torch.int32, base=m.torch.int32)
                    if family == 'direct':
                        del expected['scales'];expected['origin'] = m.torch.int32
                    tensors = {k: v for k, v in c.items() if isinstance(v, m.torch.Tensor)}
                    assert set(tensors) == set(expected), (role, tensors.keys())
                    assert all(v.dtype == expected[k] for k, v in tensors.items()), role
                    # FP32 scales encode one exponent-window origin per 16 rows, not expanded weights.
                    if family == 'direct':
                        assert not bool((c['base'] & 1).any()), role
                        assert m.torch.equal(c['origin'], (c['base'] >> 1) * 0x01010101), role
                    else:
                        assert m.torch.equal(c['scales'], m.torch.pow(2.0, (c['base'] - 112).double()).float()), role
                    assert bool(((c['esc'][:, 1] >= 0) & (c['esc'][:, 1] <= 65535)).all()), role
                    codes[str(key)] = {k: dict(dtype=str(v.dtype), shape=list(v.shape), bytes=v.numel() * v.element_size())
                                   for k, v in c.items() if isinstance(v, m.torch.Tensor)}
                return state

            m.prepare = checked_prepare
        return m
    local_eval.load_module = loader
    sys.argv = ['audit.py', '--candidate', str(cand), '--out', str(out)] + (['--forced'] if forced else [])
    try:
        audit.main()
    finally:
        (out / ('coded_calls_forced.json' if forced else 'coded_calls.json')).write_text(json.dumps(counts, indent=2) + '\n')
        (out / 'coded_state.json').write_text(json.dumps(codes, indent=2) + '\n')
        print('CODED_CALLS', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
