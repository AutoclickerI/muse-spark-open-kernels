#!/usr/bin/env python3
"""Shared lossless V9 generators. CLI builds the current seed lab and drop-in."""
import ast
import hashlib
import json
from pathlib import Path
from build_moe_real_routing import namespace,ROOT,HERE
REUSE=ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/reuse.py'


def literal(source,name):
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
    raise KeyError(name)


def raw_v9(source):
    """Preserve the old V9 schedule but move the E4M3 conversion into timed CUDA."""
    reuse=REUSE.read_text();cu=literal(reuse,'REUSE_CUDA')
    helper=cu[cu.index('// Raw E4M3'):cu.index('__device__ __forceinline__ uint32_t mul2')]
    source=source.replace('struct PrefetchState {',helper+'\nstruct PrefetchState {')
    source=source.replace('const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;',
                          'const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);')
    source=source.replace('const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;',
                          'const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);')
    # Replace both source variants' prepare-only scale converters with a byte layout repack.
    a=source.index('\ndef _encode_scales');b=source.index('\ndef _repack',a)
    source=source[:a]+source[b:]
    a=source.index('    u, neg, zero =');b=source.index('    codes = codes.reshape',a)
    source=source[:a]+'    u = sf.view(torch.uint8)  # E4M3 bytes stay E4M3\n'+source[b:]
    return source


def wide_v9(source):
    """Expose V9 NT=2/4 without changing its activation reuse or tile-stream schedule."""
    source=source.replace('__launch_bounds__(NTHR, 2) moe_fused','__launch_bounds__(NTHR, NT == 1 ? 2 : 1) moe_fused')
    source=source.replace('  LAUNCH(1);','  if(nt==1){LAUNCH(1);}else if(nt==2){LAUNCH(2);}else if(nt==4){LAUNCH(4);}else{TORCH_CHECK(false,"V9 NT must be 1/2/4");}')
    source=source.replace('C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));',
        'C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));')
    old='C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<1>, NTHR, smem_bytes(nt,nstage,M)));'
    source=source.replace(old,'if(nt==1){'+old+'}else if(nt==2){'+old.replace('moe_fused<1>','moe_fused<2>')+'}else{'+old.replace('moe_fused<1>','moe_fused<4>')+'}')
    source=source.replace('    nt = 1\n','    nt = state.get("nt",1)\n')
    source=source.replace('      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);','')
    source=source.replace('if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3,',
                          'for (int z=tid;z<8*NB;z+=NTHR) {\n          const int i = z & 7, n = z >> 3,')
    source=source.replace('if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4,',
                          'for (int z=tid;z<16*NB;z+=NTHR) {\n          const int i = z & 15, n = z >> 4,')
    source=source.replace('float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;',
                          'const float fw0=1.f+__bfloat162float(pnw[m*16+i]); float t2 = ok ? (v*fw0)*(v*fw0) : 0.f;')
    return source


def reference_generator():
    # Inline our copy of the public model generator. Never import reference on the judge.
    p=ROOT/'problems/muse_decode_layer/reference.py';s=p.read_text();tree=ast.parse(s)
    wanted={'generate_weights','_stream_basis','_router'}
    parts=['import math\nimport functools\nimport torch\n']
    constants={'HIDDEN':8192,'HEAD_DIM':64,'HQ':16,'HKV':2,'Q_SIZE':1024,'KV_SIZE':128,'QKV_SIZE':2304,
               'D_EXPERT':4096,'LOCAL_EXPERTS':32,'N_EXPERTS':256,'STREAM_SEED':5354,'STREAM_RANK':4,
               'ROUTER_STD':.045,'ROUTER_ALIGN':.07,'ROUTER_BIAS_STD':.055,'BLOCK':16,'E4M3_MAX':448.,'SCALE_LOG2_MEAN':7.,'SCALE_LOG2_STD':.4}
    parts.extend(f'{k}={v!r}\n' for k,v in constants.items())
    nv=(ROOT/'problems/muse_moe_w4a16/nvfp4.py').read_text()
    for n in ast.parse(nv).body:
        if isinstance(n,ast.FunctionDef) and n.name=='random_weights':parts.append(ast.unparse(n)+'\n')
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in wanted:parts.append(ast.unparse(n)+'\n')
    return '# Public generator sha256 '+hashlib.sha256(p.read_bytes()).hexdigest()+'\n'+namespace(''.join(parts),'rg_')



if __name__=='__main__':
    from build_moe_seed_lab import main
    main()
