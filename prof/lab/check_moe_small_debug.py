"""Exercise the exact small-M bank, native arithmetic, and forced Hopper glue locally."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'problems/muse_moe_w4a16'))
from nvfp4 import random_weights
from build_moe_small_debug import select


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def forced_glue(mod, bank):
    calls = []
    device = bank['w13p'].device
    class Portable:
        def __init__(self, capacity):
            self.smem_optin = capacity
        def max_smem_attr(self):
            return self.smem_optin
        def smem_needed(self, nt, stages, m):
            return stages * 36864 + 8192 * nt + 2048 + 40 * m
        def resident_blocks(self, nt, stages, m):
            assert self.smem_needed(nt, stages, m) <= self.smem_optin
            return 2 if stages == 1 else 1
        def run(self, *a):
            assert len(a) in (17, 19)
            x, ids, tw, pnw, w13, w2, gs13, gs2, inter, y, ss, cnt, out = a[:13]
            start, nt, stages, grid = a[13:17]
            assert (w13 is bank['w13p'] and w2 is bank['w2p'] and pnw is bank['pnw'])
            assert gs13 is bank['gs13'] and gs2 is bank['gs2']
            assert x.dtype == torch.bfloat16 and ids.dtype == torch.int32 and tw.dtype == torch.float32
            assert ids.is_contiguous() and tw.is_contiguous() and x.is_contiguous()
            assert inter.shape == y.shape == (x.shape[0] * 8, 4096)
            assert ss.shape == (x.shape[0] * 8,) and cnt.dtype == torch.int32
            assert start in (0, 96) and nt in (1, 2, 4) and grid == 132 * self.resident_blocks(nt, stages, x.shape[0])
            if len(a) == 19:
                assert a[-2:] == (1, 0)
            out.copy_(x)
            calls.append((nt, stages, grid))
    original = mod.load_inline
    original_cpp = torch.utils.cpp_extension.load_inline
    original_hopper = mod.IS_HOPPER
    try:
        mod.IS_HOPPER = True
        for capacity in (100000, 227328):
            portable = Portable(capacity)
            def fake_load(**kwargs):
                assert '-gencode=arch=compute_90a,code=sm_90a' in kwargs['extra_cuda_cflags']
                return portable
            mod.load_inline = fake_load
            torch.utils.cpp_extension.load_inline = fake_load
            for name, factory in (('_V9_MOD', mod._v9_module), ('_BEST5_NT2_MOD', mod._best5_nt2_module),
                                  ('_SMALL_NT4_MOD', mod._small_nt4_module), ('_HOT_NT1_MOD', mod._hot_nt1_module)):
                old = getattr(mod, name)
                setattr(mod, name, None)
                try:
                    assert factory() is portable
                finally:
                    setattr(mod, name, old)
            b = dict(bank, mod=portable, grid=132)
            state = dict(moe_bank=b, moe_nt2_mod=portable, moe_nt4_mod=portable, moe_hot_nt1_mod=portable)
            # Exercise the candidate's dispatch, rather than a rewritten equivalent.
            code = select((ROOT / 'problems/muse_decode_layer/cand_r3_best6_small_nt2.py').read_text(), {'_best6_call'})
            exec(compile(code, '<forced_small_dispatch>', 'exec'), mod.__dict__)
            for m in (1, 8, 16, 32):
                for start in (0, 96):
                    # Strided inputs force all dtype/contiguity adapters.
                    x = torch.randn(m, 8192, device=device).bfloat16()[:, ::2]
                    ids = torch.full((m, 16), start + 3, device=device, dtype=torch.int64)[:, ::2]
                    tw = torch.ones(m, 16, device=device, dtype=torch.float64)[:, ::2] / 8
                    d = dict(hidden_states=x, topk_ids=ids, topk_weights=tw,
                             config=dict(local_start=start, local_count=32, d_hidden=4096, d_expert=4096))
                    for choice in ('v9nt1', 'v9nt2', 'v9nt4', 'v9m1', 'hotnt1'):
                        out = mod._best6_call(d, state, choice)
                        assert torch.equal(out, x)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            fresh = mod._best6_call(d, state, choice)
                        graph.replay()
                        torch.cuda.synchronize()
                        assert fresh.shape == x.shape
                        assert torch.equal(fresh, x)
            # Shared-memory rejection must happen before any native launch.
            tiny = Portable(1)
            for call in (mod.v9_kernel_fn, mod.v9_small_nt1_kernel_fn, mod.v9_nt4_kernel_fn, mod.hot_nt1_kernel_fn):
                before = len(calls)
                try:
                    call(d, dict(b, mod=tiny))
                except ValueError:
                    pass
                else:
                    raise AssertionError('missing shared-memory rejection')
                assert len(calls) == before
        torch.cuda.synchronize()
    finally:
        mod.load_inline = original
        torch.utils.cpp_extension.load_inline = original_cpp
        mod.IS_HOPPER = original_hopper
    return len(calls)


def main():
    started = time.monotonic()
    mod = load(ROOT / 'problems/muse_prefill_layer/lab/lab_small_m_nt4.py', 'small_debug')
    gen = torch.Generator(device='cuda').manual_seed(6001)
    w = {}
    for name, n in (('w13', 8192), ('w2', 4096)):
        q, sf, gs = random_weights(n, 4096, generator=gen, device='cuda', leading=(1,))
        w[name] = q.expand(32, -1, -1)
        w[name + '_sf'] = sf.expand(32, -1, -1)
        w[name + '_gs'] = gs * torch.exp(.25 * torch.randn(32, device='cuda', generator=gen))
    w['post_norm_w'] = (.1 * torch.randn(4096, device='cuda', generator=gen)).bfloat16()
    bank = mod.v9_prepare(w)
    small = mod._small_nt4_module()
    good, message = mod.sm_bank_check(w, bank)
    assert good, message
    print(message, flush=True)
    calls = forced_glue(mod, bank)
    print('FORCED_HOPPER_GLUE_PASS', calls, flush=True)
    rows = []
    for m, nt in ((1, 1), (8, 1), (16, 4), (32, 4)):
        ids = torch.full((m, 8), 255, device='cuda', dtype=torch.int32)
        ids[:, 0] = 3
        ids[:, 1] = 7
        ids[::2, 2] = 11
        data = dict(hidden_states=torch.randn(m, 4096, device='cuda', generator=gen).bfloat16(), topk_ids=ids,
                    topk_weights=torch.softmax(torch.randn(m, 8, device='cuda', generator=gen), -1),
                    config=dict(local_start=0, local_count=32, d_hidden=4096, d_expert=4096))
        base = mod.sm_run_scratch(data, bank, bank['mod'], 1)
        test = mod.sm_run_scratch(data, bank, small, nt)
        _, status, first, message = mod.sm_stage_check(data, w, base, test, f'NT{nt}_M{m}')
        print(message, first, flush=True)
        assert status == 'PASS', first
        graph = mod.LabGraph(lambda d, nt=nt: mod.sm_small_call(d, dict(bank, mod=small), nt), data)
        for j in range(3):
            d = dict(data, hidden_states=torch.randn(m, 4096, device='cuda', generator=gen).bfloat16())
            graph.refresh(d)
            got = graph.replay()
            ref = mod.sm_run_scratch(d, bank, bank['mod'], 1)[3]
            rel, row = mod.sm_error(got, ref)
            assert rel < .01 and row < .05, (m, nt, rel, row)
        rows.append(message)
        if m == 1:
            # Reproduce the old lab's exact bad pairing, then locate its first wrong stage.
            legacy = load(ROOT / 'problems/muse_moe_w4a16/cand_v9.py', 'legacy_v9')
            bad = dict(bank, w13p=bank['w13p'].clone(), w2p=bank['w2p'].clone())
            for expert in (3, 7, 11):
                t = torch.arange(512, device='cuda')[:, None] * 8 + torch.arange(8, device='cuda')
                rowmap = torch.cat((t, 4096 + t), -1)
                bad['w13p'][expert] = legacy._repack(w['w13'][expert], w['w13_sf'][expert], rowmap)
                bad['w2p'][expert] = legacy._repack(w['w2'][expert], w['w2_sf'][expert], torch.arange(4096, device='cuda').view(256, 16))
            broken = mod.sm_run_scratch(data, bad, small, 1)
            _, status, first, message = mod.sm_stage_check(data, w, base, broken, 'old_bank_raw_reader')
            print('EXPECTED_REPRODUCTION', message, first, flush=True)
            assert status == 'FAIL' and 'swiglu' in first
            rel, _ = mod.sm_error(broken[3], base[3])
            assert rel > .5, rel
    result = dict(device=torch.cuda.get_device_name(), forced_calls=calls, native_checks=rows,
                  old_bank_mismatch_reproduced=True, seconds=time.monotonic() - started)
    out = ROOT / 'runs/screen/moe_small_debug.json'
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
