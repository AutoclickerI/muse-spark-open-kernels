"""Build one standalone, raw-scale decode-shape adapter comparison lab."""
import ast
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'problems/muse_moe_w4a16/dev/mixed_wgmma'))
from build_prefill_lab import subset


def replace_once(src, old, new):
    assert src.count(old) == 1, (old[:100], src.count(old))
    return src.replace(old, new)


def main():
    seed_path = ROOT / 'prof/lab/moe_v9_seed.py'
    pref_path = ROOT / 'prof/lab/moe_mixed_wgmma_prefill.py'
    seed = seed_path.read_text()
    pref = pref_path.read_text()
    native = seed[:seed.index('\n"""NVFP4 BF16 MoE:', seed.index('def rv_kernel_fn'))]
    # The native bank stores raw E4M3 bytes. Only the W13 epilogue gains
    # independent gate/up FP32 GS (first/second 32-element halves).
    tree = ast.parse(native)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rv_CUDA_SRC' for t in node.targets):
            cuda = ast.literal_eval(node.value)
            cuda = replace_once(cuda, 'U *= sc;', 'U *= gs13[e + NLOC] * 256.f;')
            node.value = ast.Constant(cuda)
    native = ast.unparse(tree)
    helpers = subset(seed, ('rg_generate_weights', 'fx_make_fixture', 'CAPTURED_R0'))
    # Drop duplicate native/hot imports from helpers; subset retains imports only,
    # not the unrequested native/hot definitions.
    control = pref[pref.index('import ctypes as pf_ct'):pref.index('class MWLocalDGStandin:')]
    control = subset(control, ('pf__grouped_module', 'pf__raw_repack', 'pf__encode_blob_fast', 'pf__raw_experts_tiled',
                              'pf__run_canonical_dg', 'pf__Experts', 'pf__experts_humming_masked'))
    tree = ast.parse(control)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pf_GROUPED_SRC' for t in node.targets):
            src = ast.literal_eval(node.value)
            # Keep just the exact raw17 producer and its existing SM120 emulation.
            header = src[src.index('#include <torch/extension.h>'):src.index('namespace canonical {')]
            src = header + '#include <cuda.h>\n#include <cuda_runtime.h>\n#define MOE_WGMMA USE_WGMMA\n' + src[src.index('namespace raw17 {'):src.index('\n\n#include <cublasLt.h>', src.index('namespace raw17 {'))]
            old = '''  uint32_t lo = two_bytes & 255, hi = (two_bytes >> 8) & 255;
  uint32_t bits = (lo ? 0x7000u + (lo << 4) : 0) | ((hi ? 0x7000u + (hi << 4) : 0) << 16);'''
            new = '''  uint32_t v[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint32_t raw = (two_bytes >> (8 * i)) & 255u;
    uint32_t mag = raw & 127u, exp = mag >> 3, m = mag & 7u, b = 0;
    if (exp) b = ((exp + 238) << 7) | (m << 4);
    else if (m) { int k = 31 - __clz(m); b = ((236 + k) << 7) | ((m << (7 - k)) & 127u); }
    v[i] = b | ((raw & 128u) << 8);
  }
  uint32_t bits = v[0] | (v[1] << 16);'''
            src = replace_once(src, old, new)
            src = replace_once(src, 'acc[1][4 * i + 2 * h + q] * gsc', 'acc[1][4 * i + 2 * h + q] * (p.gs[e + 32] * 256.f)')
            node.value = ast.Constant(src)
        if isinstance(node, ast.FunctionDef) and node.name == 'pf__grouped_module':
            body = '''
def pf__grouped_module():
    global pf__mod
    if pf__mod is None:
        from torch.utils.cpp_extension import load_inline
        cap = pf_torch.cuda.get_device_capability()
        previous = pf_os.environ.get('TORCH_CUDA_ARCH_LIST')
        pf_os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if cap == (9, 0) else f'{cap[0]}.{cap[1]}'
        try:
            pf__mod = load_inline(name='decode_raw17_' + pf_hashlib.sha256(pf_GROUPED_SRC.encode()).hexdigest()[:12], cpp_sources='void raw17_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);', cuda_sources=pf_GROUPED_SRC, functions=['raw17_gemm'], extra_cuda_cflags=['-O3', '-std=c++20'], verbose=False)
        finally:
            if previous is None: pf_os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else: pf_os.environ['TORCH_CUDA_ARCH_LIST'] = previous
    return pf__mod
'''
            node.body = ast.parse(body).body[0].body
        if isinstance(node, ast.FunctionDef) and node.name == 'pf__encode_blob_fast':
            # Only permute the E2M1 bits. Do not fold scale signs or zeros into codes.
            code = ast.unparse(node)
            import re
            code = re.sub(r' \^ pf_tl.where\(s\d & 128 != 0, 8, 0\).to\(pf_tl.uint32\)', '', code)
            code = re.sub(r'(?m)^        c(\d+) = pf_tl.where\(s\d & 127 != 0, c\1, 0\)\n', '', code)
            # Remove scale loads; scale bytes are a separate unchanged bank.
            code = re.sub(r'(?m)^        s\d = pf_tl.load\([^\n]+\n', '', code)
            node.body = ast.parse(code).body[0].body
        if isinstance(node, ast.FunctionDef) and node.name == 'pf__canonical_expand_coalesced':
            code = ast.unparse(node)
            code = replace_once(code, 'scale = 1879076864 + sf * 1048592', '''sval = sf.to(pf_tl.uint8).to(pf_tl.float8e4nv, bitcast=True).to(pf_tl.float32) * (2.0 ** 118)
    sbits = sval.to(pf_tl.bfloat16).to(pf_tl.uint16, bitcast=True).to(pf_tl.uint32)
    scale = sbits | (sbits << 16)''')
            node.body = ast.parse(code).body[0].body
        if isinstance(node, ast.ClassDef) and node.name == 'pf__Experts':
            # Do not compile unused prefill/indexed plans. Masked 128/256 plans
            # are compiled by dc_prepare after setting GROUPED_MASKED.
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == '__init__':
                    fn.body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == 'compile')]
    control = ast.unparse(ast.fix_missing_locations(tree))
    # Humming gets unit W13 GS, preserving the E4M3 block-scale representation;
    # apply distinct gate/up FP32 GS in the existing portfolio SwiGLU epilogue.
    control = replace_once(control, "pf__swiglu_kernel[32, pf_triton.cdiv(T, 4)](h, inter, starts, counts, st['ones'], BR=4, D=4096, num_warps=8)", "pf__portfolio_swiglu[32, pf_triton.cdiv(T, 4)](h, inter, starts, counts, st['gs13_gate'], st['gs13_up'], BR=4, D=4096, num_warps=8)")
    route = next(n for n in ast.parse(pref).body if isinstance(n, ast.FunctionDef) and n.name == 'mw_route')
    portfolio = next(n for n in ast.parse(pref).body if isinstance(n, ast.FunctionDef) and n.name == 'pf__portfolio_swiglu')
    graph = next(n for n in ast.parse(pref).body if isinstance(n, ast.ClassDef) and n.name == 'MWGraph')
    out = ROOT / 'prof/lab/moe_decode_adapters.py'
    metadata = '\n'.join(f'# Source {p.relative_to(ROOT)} SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}' for p in (seed_path, pref_path))
    out.write_text('"""Decode-shape FP32-gated raw17/masked-Humming/DG vs native V9 NT1/NT2. Generated by decode_adapters/build.py."""\n' + metadata + '\n' + native + '\n' + helpers + '\nimport triton\nimport triton.language as tl\n' + ast.unparse(route) + '\n' + control + '\n' + ast.unparse(portfolio) + '\n' + ast.unparse(graph) + '\n' + (Path(__file__).with_name('driver.py.inc')).read_text())
    print(out.relative_to(ROOT), hashlib.sha256(out.read_bytes()).hexdigest(), out.stat().st_size)


if __name__ == '__main__':
    main()
