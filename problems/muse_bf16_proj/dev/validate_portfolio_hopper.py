"""Reach every configuration emitted by the actual Hopper portfolio enumerator.

Metadata tensors plus explicit, signature-checked portable entrypoints exercise
the real Python dispatcher without running Hopper instructions on sm_120.
The archived failed source must reproduce its missing old_hopper entrypoint.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(path):
    spec = importlib.util.spec_from_file_location('portfolio_hopper_coverage', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._hopper = lambda: True
    return module


class Portable:
    # Deliberately no __getattr__, *args or old_hopper: absent entrypoints and
    # wrong tuple lengths must fail just as they do in the actual native ABI.
    def skinny(self, w, lo, x, out, n, nt, tr, u, wn, wk, ks, split2, xp, ld):
        assert out.shape == (x.shape[0], n) and w.shape[1] == x.shape[1]
        assert n % (wn * tr * 16) == 0 and x.shape[1] % (wk * 32 * u * ks) == 0
        assert split2 == (out.dtype == torch.float32)
        out.zero_()

    def hopper(self, w, x, out, work, counts, bt, bk, st, mode, split, threads):
        assert threads in (128, 160, 256) and mode in (0, 1, 2, 3)
        assert w.shape[1] == x.shape[1] * (2 if out.dtype == torch.float32 else 1)
        assert work.dtype == torch.float32 and counts.dtype == torch.uint32
        out.zero_()

    def lt_run(self, w, x, out, work, codes, index):
        assert 0 <= index < len(codes) and w.shape[0] >= out.shape[1] and w.shape[1] == x.shape[1]
        out.zero_()

    def all_lt(self, w, x, out, work, codes, index, nn, kind, accumulate=False):
        assert 0 <= index < len(codes) and nn in (0, 1) and kind in (0, 1, 2)
        assert (w.shape[0] == x.shape[1] and w.shape[1] >= out.shape[1]) if nn else (w.shape[0] >= out.shape[1] and w.shape[1] == x.shape[1])
        assert x.dtype == (torch.float32 if kind == 2 else torch.bfloat16)
        assert not accumulate or kind == 1
        out.zero_()

    def gemv(self, w, x, out, variant):
        assert x.shape[0] == 1 and variant in range(4)
        out.zero_()

    def tc(self, w, x, out, bt, kt):
        assert bt in (16, 32) and kt in (128, 256)
        out.zero_()

    def m16(self, w, x, out):
        assert x.shape[0] == 16
        out.zero_()


def dsv3(x, w, output):
    assert x.shape[0] == 16 and w.shape == (x.shape[1], output.shape[1])
    output.zero_()


def exercise(c, path):
    # These are the actual problem dimensions, including the unpooler slice.
    shapes = dict(qkv_gate=(2304, 8192), o=(8192, 1024),
                  pooler=(4096, 8192), unpooler=(8192, 4096), router=(256, 8192))
    state = dict(mod=Portable(), scratch={}, codes={}, legacy_codes={},
                 transposed={}, workspace=torch.empty(512 << 20, device='meta', dtype=torch.uint8),
                 legacy_workspace=torch.empty(32 << 20, device='meta', dtype=torch.uint8), dsv3_fn=dsv3)
    for layer, (n, k) in shapes.items():
        state[layer] = torch.empty((n, k), device='meta', dtype=torch.float32 if layer == 'router' else torch.bfloat16)
        state['transposed'][layer] = state[layer].T.contiguous()
    state['router_hi'] = state['router'].to(torch.bfloat16)
    state['router_lo'] = state['router_hi'].clone()
    for suffix in ('hi', 'lo', 'fp32'):
        state['router_' + suffix] = state['router'] if suffix == 'fp32' else state['router_' + suffix]
        state['router_' + suffix + '_t'] = state['router_' + suffix].T.contiguous()
    state['router_pair'] = torch.cat((state['router_hi'], state['router_lo']), 1)
    def incumbent(data, st):
        cfg = data['config']
        return torch.empty((data['x'].shape[0], cfg['n']), device='meta',
                           dtype=torch.float32 if cfg['layer'] == 'router' else torch.bfloat16)
    c._small = incumbent
    props = SimpleNamespace(multi_processor_count=132, shared_memory_per_block_optin=227328)
    families, subfamilies = Counter(), Counter()
    configurations = 0
    for layer, (full_n, k) in shapes.items():
        for m in (1, 8, 16, 32, 128, 256):
            n = 1024 if layer == 'unpooler' and m <= 8 else full_n
            key = (layer, m, n, k)
            state['scratch'][key] = (torch.empty((1,), device='meta'), torch.zeros((1,), device='meta', dtype=torch.uint32))
            state['legacy_codes'][key] = torch.empty((32, 16), dtype=torch.uint8)
            for nn in (0, 1):
                for kind in (0, 1, 2):
                    state['codes'][(key, nn, kind)] = torch.empty((2, 16), dtype=torch.uint8)
            x = torch.empty((m, k), device='meta', dtype=torch.bfloat16)
            data = dict(x=x, config=dict(layer=layer, m=m, n=n, k=k))
            for config in c._all_variants(state, layer, m, n, k, props):
                # Do not substitute the enumerator or dispatcher under test.
                out = c._all_run(data, state, config)
                assert out.shape == (m, n) and out.dtype == (torch.float32 if layer == 'router' else torch.bfloat16)
                families[config[0]] += 1
                sub = (config[0], layer == 'router')
                if config[0] == 'pair': sub += (config[4], config[-1])
                elif config[0] in ('lt', 'router_lt', 'router_fp32'): sub += (config[1],)
                subfamilies[str(sub)] += 1
                configurations += 1
    expected = {'incumbent', 'legacy_lt', 'torch', 'gemv', 'm16', 'dsv3', 'tc', 'skin', 'pair', 'lt', 'router_lt', 'router_fp32'}
    assert set(families) == expected, families
    return dict(source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                configurations=configurations, families=dict(families), subfamilies=dict(subfamilies))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    path = ROOT / 'problems/muse_bf16_proj/cand_portfolio_all.py'
    report = exercise(load(path), path)
    with tempfile.TemporaryDirectory() as tmp:
        archived = Path(tmp) / 'failed_portfolio.py'
        archived.write_bytes(subprocess.check_output(['git', 'show',
            '1d6a356:problems/muse_bf16_proj/cand_portfolio_all.py'], cwd=ROOT))
        try:
            exercise(load(archived), archived)
        except AttributeError as exc:
            assert 'old_hopper' in str(exc), exc
            report['failed_source_regression_reproduced'] = True
        else:
            raise AssertionError('the failed archived source unexpectedly passed coverage')
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
