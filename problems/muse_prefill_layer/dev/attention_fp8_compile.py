"""Compile the actual attention source for Hopper without executing Hopper instructions."""
import importlib.util
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
p = Path('problems/muse_prefill_layer/cand_attention_fp8.py')
s = importlib.util.spec_from_file_location('attn_fp8_candidate', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
sig = dict(q_ptr='*fp8e4nv', kc_ptr='*fp8e4nv', vc_ptr='*fp8e4nv', pt_ptr='*i32',
           seq_ptr='*i32', o_ptr='*bf16', ks_ptr='*fp32', vs_ptr='*fp32',
           scale='fp32', L='i32', max_pages='i32')
root = Path('problems/muse_prefill_layer/dev')
for contig,bn in ((False,128),(True,256)):
    const = dict(PAGE=1, WINDOW=0, BQ=16, BN=bn, GROUP=8, PSCALE=448.0, CONTIG=contig)
    for key in const:
        sig[key] = 'constexpr'
    compiled = triton.compile(ASTSource(fn=m._attn_fp8_kernel, signature=sig, constexprs=const),
                              target=GPUTarget('cuda', 90, 32), options=dict(num_warps=8, num_stages=2))
    (root / ('attention_fp8_contig_sm90.ptx' if contig else 'attention_fp8_sm90.ptx')).write_text(compiled.asm['ptx'])
    (root / ('attention_fp8_contig_sm90.cubin' if contig else 'attention_fp8_sm90.cubin')).write_bytes(compiled.asm['cubin'])
    print('SM90 compile PASS', 'contiguous',contig,compiled.metadata)
    ptx=compiled.asm['ptx']
    print('wgmma FP8 instructions', sum('wgmma.mma_async' in x and 'e4m3' in x for x in ptx.splitlines()))
    print('mma FP8 instructions', sum('mma.sync' in x and 'e4m3' in x for x in ptx.splitlines()))
