"""Freeze paired raw/recoded M128 producers and the actual board NT1 reader."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
M128 = HERE.parent
ROOT = HERE.parents[5]
sys.path[:0] = [str(M128), str(M128.parent), str(ROOT / 'tools')]
from build_prefill_lab import subset
from build_repair_lab import rename, pointer_bindings
from native_abi import CPP_PRELUDE
from source import source, CONFIGS
from okc_run import rejected_words


def recode_source(raw):
    a = raw.index('__device__ __forceinline__ unsigned sf_bits(')
    b = raw.index('__device__ __forceinline__ unsigned mul2(', a)
    raw = raw[:a] + raw[b:]
    raw = raw.replace(',const unsigned*lut,', ',')
    old = 'unsigned s0=lut[raw&255],s1=lut[(raw>>8)&255],s2=lut[(raw>>16)&255],s3=lut[raw>>24];'
    new = '''unsigned s0=__byte_perm(raw,0u,0x4040u)*16u+0x70007000u;
 unsigned s1=__byte_perm(raw,0u,0x4141u)*16u+0x70007000u;
 unsigned s2=__byte_perm(raw,0u,0x4242u)*16u+0x70007000u;
 unsigned s3=__byte_perm(raw,0u,0x4343u)*16u+0x70007000u;'''
    assert old in raw
    raw = raw.replace(old, new).replace(',lut,', ',')
    raw = raw.replace('__shared__ unsigned lut[256];', '')
    old = 'lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();'
    assert old in raw
    raw = raw.replace(old, '')
    assert 'lut' not in raw and 'sf_bits' not in raw
    return raw


def main():
    raw = source()
    recoded = recode_source(raw)
    native = '\n'.join(pointer_bindings(rename(body, 'rot64', (3, 4, 5))).replace('rot64', prefix)
                       for prefix, body in (('msraw', raw), ('msrec', recoded)))
    cpp = CPP_PRELUDE + '\n'.join(
        f'int64_t {p}_init();std::vector<int64_t> {p}_info(int64_t,int64_t);'
        f'void {p}_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);'
        f'void {p}_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'
        for p in ('msraw', 'msrec'))
    funs = [p + '_' + f for p in ('msraw', 'msrec') for f in ('init', 'info', 'route', 'run')]
    helpers = subset((ROOT / 'prof/lab/moe_m128_k64_repair.py').read_text(),
                     ('dc_weights', 'dc_fixtures', 'dc_oracles', 'MWGraph', 'mw_repack_planes', 'repair_plan'))
    # Drop all old native constants/builders accidentally reached by the plan's
    # closures; the plan itself only calls the bank's pointer ABI.
    assert 'REPAIR_CUDA' not in helpers and 'RepairABI' not in helpers
    board = ROOT / 'problems/muse_decode_layer/cand_r3_v9.py'
    nt1 = subset(board.read_text(), ('v9_prepare', 'v9_kernel_fn'))
    # The actual reader and packer are kept; unrelated imports do not belong in
    # this payload or its local forced-Hopper screen.
    tree = ast.parse(nt1)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Try))]
    nt1 = ast.unparse(tree)
    text = ('"""M128 exact scale-bank producer comparison, independent public router draws."""\n'
            + helpers + '\n' + nt1 + '\nimport functools\nimport gc\n'
            + 'from torch.utils.cpp_extension import load_inline\n'
            + 'CONFIGS=' + repr(CONFIGS) + '\nSCALE_CUDA=' + repr(native)
            + '\nSCALE_CPP=' + repr(cpp) + '\nSCALE_FUNCTIONS=' + repr(funs)
            + '\n' + (HERE / 'driver.py.inc').read_text())
    assert not rejected_words(text), rejected_words(text)
    out = ROOT / 'prof/lab/moe_m128_scale_bank.py'
    out.write_text(text)
    manifest = dict(payload=str(out.relative_to(ROOT)), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    native_sha256=hashlib.sha256(native.encode()).hexdigest(),
                    source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    recoded_source_sha256=hashlib.sha256(recoded.encode()).hexdigest(),
                    nt1_source=str(board.relative_to(ROOT)), nt1_sha256=hashlib.sha256(board.read_bytes()).hexdigest(),
                    configs=[3, 4, 5], grids=[528], weight_seeds=[5353 + 7919 * i for i in range(12)],
                    inputs_per_seed=3, rounds=6, cold_bytes=128 << 20)
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out.relative_to(ROOT), manifest['sha256'], out.stat().st_size)


if __name__ == '__main__':
    main()
