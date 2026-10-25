"""S9 normal local_eval plus the prepare() audit of the new cells' tensors and their dispatch.

After prepare: every tensor the small-M cell (s4_nt1_bank) and N_full (s5_hh_base, moe_bank, nf_streams) keep for
the timed call, with dtype, shape, bytes and, for expert banks, bytes per 16 weights (NVFP4 limit: <= 9 B with
4-bit codes and 8-bit scales). During the run: smallm_experts calls by (M, flags) and nf_call calls, eager and
captured. Writes audit.json, normal_local.json.

    CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/s9/audit.py [--forced] [--cold-native]
        [--candidate cand.py --out dev/<dir>]
    Cold prepare: fresh TORCH_EXTENSIONS_DIR / TRITON_CACHE_DIR / CUDA_CACHE_PATH and CCACHE_DISABLE=1 (the image's
    extension builds go through ccache).
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
sys.path.insert(0, str(R / 'tools'))
import local_eval  # noqa: E402

CAND = R / 'problems/muse_decode_layer/cand_r3_best7_s9.py'


def inventory(torch, state, weights):
    """name -> dtype, shape, bytes, (expert banks) bytes per 16 weights of the matrix it encodes."""
    groups = {k: weights[k].numel() * 2 // 16 for k in ('w13', 'w2')}  # FP4 codes: two weights per byte
    rows, seen = {}, {}

    def walk(prefix, v):
        if isinstance(v, torch.Tensor):
            key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape))
            row = dict(dtype=str(v.dtype).removeprefix('torch.'), shape=list(v.shape), bytes=v.numel() * v.element_size())
            if key in seen:row['alias_of'] = seen[key]
            else:seen[key] = prefix
            last = prefix.rsplit('.', 1)[-1]
            mat = 'w13' if last.startswith('w13') else 'w2' if last.startswith('w2') else None
            if mat and '_gs' not in last:
                row['matrix'] = mat
                row['bytes_per_16'] = row['bytes'] / groups[mat]
            rows[prefix] = row
        elif isinstance(v, dict):
            for k, x in v.items():walk(f'{prefix}.{k}', x)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):walk(f'{prefix}[{i}]', x)
        else:
            rows[prefix] = dict(type=type(v).__name__)

    for name in ('s4_nt1_bank', 's5_hh_base', 'moe_bank', 'nf_streams'):walk(name, state[name])
    per = {}
    for name, row in rows.items():
        if 'matrix' in row and 'alias_of' not in row:
            bank = name.split('.')[0]
            per.setdefault(f"{bank}:{row['matrix']}", 0.)
            per[f"{bank}:{row['matrix']}"] += row['bytes_per_16']
    return rows, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forced', action='store_true', help='force the Hopper glue (best7/force.py) and require native S9 math per row')
    ap.add_argument('--cold-native', action='store_true')
    ap.add_argument('--candidate', type=Path, default=CAND)
    ap.add_argument('--out', type=Path, default=H, help='directory for the json outputs')
    a = ap.parse_args()
    cand, out_dir = a.candidate.resolve(), a.out.resolve()
    original = local_eval.load_module
    records = dict(calls={}, nf={})
    force = None
    if a.forced:
        spec = importlib.util.spec_from_file_location('best7_force', H.parent / 'best7/force.py')
        force = importlib.util.module_from_spec(spec);spec.loader.exec_module(force)

    def loader(name, path):
        m = original(name, path)
        if name != 'submission':return m
        if force:records['forced'] = force.install(m, a.cold_native, False)
        torch = m.torch
        import threading
        import time
        t0 = time.monotonic()
        builds = records.setdefault('native_builds', [])

        def timed(fn):
            def build(*args, **kw):
                start = time.monotonic()
                try:return fn(*args, **kw)
                finally:builds.append(dict(name=kw.get('name'), start_s=round(start - t0, 2), seconds=round(time.monotonic() - start, 2),
                                           thread=threading.current_thread().name))
            return build
        for alias in ('load_inline', '_sp_load_inline', '_p7load', '_gfload'):
            if hasattr(m, alias):setattr(m, alias, timed(getattr(m, alias)))
        import torch.utils.cpp_extension as ext  # builders that import load_inline inside the function
        ext.load_inline = timed(ext.load_inline)
        small, nf = m.smallm_experts, m.nf_call

        def counted_small(data, state, flags, stages=2, grid=264):
            mm = data['hidden_states'].shape[0]
            assert flags == m.SMALLM_FLAGS[mm] and (stages, grid) == (2, 264), (mm, flags, stages, grid)
            row = records['calls'].setdefault(f'M{mm}:{flags}', dict(eager=0, captured=0))
            row['captured' if torch.cuda.is_current_stream_capturing() else 'eager'] += 1
            return small(data, state, flags, stages, grid)

        def counted_nf(data, state):
            mm = data['hidden_states'].shape[0]
            assert mm in (128, 256), mm
            row = records['nf'].setdefault(f'M{mm}', dict(eager=0, captured=0))
            row['captured' if torch.cuda.is_current_stream_capturing() else 'eager'] += 1
            return nf(data, state)
        records['smallm_flags'] = dict(m.SMALLM_FLAGS)
        m.smallm_experts = counted_small
        m.nf_call = counted_nf
        prepared = m.prepare

        def prepare(weights):
            state = prepared(weights)
            rows, per = inventory(torch, state, weights)
            records.update(inventory=rows, bytes_per_16_weights=per)
            assert all(v <= 9.0 + 1e-9 for v in per.values()), per
            plans = {str(k): p['moe'] for k, p in state['best7_plans'].items() if k != 'other'}
            records['plans'] = plans
            records['gates'] = {str(k): v for k, v in state['best7_gates'].items() if k[0] == 'moe'}
            return state
        m.prepare = prepare
        return m

    local_eval.load_module = loader
    out = out_dir / ('forced_local.json' if a.forced else 'normal_local.json')
    sys.argv = [sys.argv[0], str(cand), '--no-baseline', '--json', str(out)]
    try:
        local_eval.main()
    except SystemExit as exc:
        if exc.code == 0:
            small_rows = [k for k, v in records['plans'].items() if v == 'nt1_lop3']
            assert len(small_rows) == 8, records['plans']
            assert sorted(records['plans'].values()).count('hh_m128') == 2 and sorted(records['plans'].values()).count('mixed_parallel_m256') == 2
            want = {f'M{mm}:{f}' for mm, f in records['smallm_flags'].items()}
            assert set(records['calls']) == want and all(v['captured'] > 0 for v in records['calls'].values()), records['calls']
            assert set(records['nf']) == {'M128', 'M256'} and all(v['captured'] > 0 for v in records['nf'].values()), records['nf']
            records['sha256'] = hashlib.sha256(cand.read_bytes()).hexdigest()
            (out_dir / ('forced_audit.json' if a.forced else 'audit.json')).write_text(json.dumps(records, indent=2, default=str) + '\n')
        raise


if __name__ == '__main__':
    main()
