"""Screen the actual MoE host wrapper and record every forced producer graph."""
from pathlib import Path
import hashlib
import json
import linecache
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(ROOT / 'tools'))
import local_eval


def encoding_gate(ns):
    raw = torch.arange(256, device='cuda', dtype=torch.uint8)
    raw = raw[(raw & 127) != 127]
    sf = raw.view(torch.float8_e4m3fn)
    code, neg, zero = ns['_encode_scales'](sf)
    # Check every finite E4M3 bit pattern against all sixteen E2M1 codes,
    # including negative/zero scales and subnormals. This also checks the
    # producer's high-exponent BF16 multiplier, without using frexp to invert it.
    c = torch.arange(16, device='cuda', dtype=torch.int32)
    lut = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device='cuda')
    original = lut[c & 7][None, :] * torch.where((c & 8) != 0, -1., 1.)[None, :] * sf.float()[:, None]
    folded = torch.where(zero[:, None], torch.zeros_like(c)[None, :], c[None, :] ^ (neg.int()[:, None] * 8))
    bits = ((code.int() << 4) + 0x7000).to(torch.int16).view(torch.bfloat16).float()
    decoded = lut[folded & 7] * torch.where((folded & 8) != 0, -1., 1.) * (bits[:, None] * 2.**-118)
    assert torch.equal(original, decoded), 'scale recode lost a finite NVFP4 value'
    return dict(finite_scale_patterns=len(raw), fp4_codes=16, exact_values=original.numel(),
                retained_bytes_per_block=9)


def main():
    original_load = local_eval.load_module
    report = {}
    def loader(name, path):
        module = original_load(name, path)
        if name == 'submission':
            def run_one(label, source, deadline, weights, state):
                filename = '<h200_lab ' + label + '>'
                linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
                ns = dict(module.__dict__); exec(compile(source, filename, 'exec'), ns)
                report['encoding'] = encoding_gate(ns)
                start = time.monotonic()
                lines = ns['lab'](min(deadline, start + 78.), weights, state)
                rows = ns['SCALE_RECORDS']
                report.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),
                              lines=lines, elapsed_s=time.monotonic()-start, rows=rows,
                              forced_producer_graphs=len(rows)*6, nt1_graphs=len(rows),
                              portable='SM120 immediate MMA with identical pointer ABI, launch glue and Hopper B128 layout')
                (HERE / 'agreement.json').write_text(json.dumps(report, indent=2) + '\n')
                assert len(rows) == 36, lines[-1]
                assert not any('INCOMPLETE' in line or 'REJECT' in line for line in lines), lines
                assert all(e <= .01 and r <= .05 for row in rows for e,r in row['errors'])
                assert all(len(row['samples'][0]) == 12 and all(len(v)==6 for v in row['samples'][1:]) for row in rows)
                assert sum(len(line)+1 for line in lines) <= 4500
                print('SCALE_LOCAL_PASS', len(rows), 'fixtures;', report['forced_producer_graphs'], 'forced graphs;',
                      report['encoding']['exact_values'], 'exact finite NVFP4 values', flush=True)
                return lines
            module._lab_run_one = run_one
        return module
    local_eval.load_module = loader
    sys.argv = [sys.argv[0], str(ROOT / 'problems/muse_moe_w4a16/lab/lab_moe_m128_scale_bank.py'),
                '--no-baseline', '--json', str(HERE / 'wrapper_screen.json')]
    try:
        local_eval.main()
    except SystemExit as exc:
        if exc.code == 0:
            assert len(report.get('rows', [])) == 36, 'payload failed despite host passing'
        raise


if __name__ == '__main__':
    main()
