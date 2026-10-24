#!/usr/bin/env python3
"""Assemble self-contained V9 tile-reuse payload and a decoder-layer drop-in."""
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


def build_lab():
    v=(ROOT/'problems/muse_moe_w4a16/cand_v9.py').read_text()
    ph=(ROOT/'problems/muse_moe_w4a16/cand_v9_phase.py').read_text().split('\nimport json')[0]
    # Keep absolute per-CTA stamps and activation reloads alongside the existing phase counters.
    ph=ph.replace('TR_STRIDE=10','TR_STRIDE=14').replace('TRACE_STRIDE = 10','TRACE_STRIDE = 14')
    ph=ph.replace('int st = 0;', 'int trace_reloads = 0;\n  int st = 0;')
    ph=ph.replace('cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;',
                  'cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0; ++trace_reloads;')
    ph=ph.replace('trace[q + TR_ATOMICS] = tr_atomics;',
                  'trace[q + TR_ATOMICS] = tr_atomics; trace[q+10]=tr_route0; trace[q+11]=clock64(); trace[q+12]=trace_reloads; trace[q+13]=rt->nblk[cur_e >= 0 ? cur_e : 0];')
    fx=(ROOT/'problems/muse_decode_layer/moe_tune.py').read_text().split('\ndef _graph_time')[0]
    h=(HERE/'moe_humming_real_routing.py').read_text();pack=literal(h,'PROBE_PACK_R')
    chunks=['"""V9 reuse lab: exact captured router IDs plus public generator seed/fields. No decode quota."""\n',
            namespace(wide_v9(raw_v9(v)),'base_'),namespace(raw_v9(ph),'phase_'),namespace(REUSE.read_text(),'rr_'),
            namespace(fx,'fx_'),reference_generator(),f'CAPTURED_R0={pack!r}\n',
            (HERE/'moe_v9_reuse_driver.py').read_text()]
    out=HERE/'moe_v9_reuse.py';out.write_text('\n'.join(chunks))
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest())


def build_candidate():
    base=ROOT/'problems/muse_decode_layer/cand_r3_v9.py'
    native=namespace(wide_v9(raw_v9((ROOT/'problems/muse_moe_w4a16/cand_v9.py').read_text())),'rv_')
    extra=namespace(REUSE.read_text(),'rr_')
    fixture=namespace((ROOT/'problems/muse_decode_layer/moe_tune.py').read_text().split('\ndef _graph_time')[0],'fx_')
    tail=(HERE/'moe_v9_reuse_dropin.py').read_text()
    tail=tail.replace('_reuse_original_prepare=prepare','').replace('_reuse_original_experts=experts_v9','')
    tail=tail.replace('def prepare(weights):','def moe_reuse_prepare(weights):')
    tail=tail.replace("state=_reuse_original_prepare(weights)","state=dict(moe_bank=rv_prepare(dict(weights,post_norm_w=weights['post_expert_norm'])))")
    tail=tail.replace('v9_kernel_fn','rv_kernel_fn').replace('def experts_v9(', 'def moe_reuse_experts(')
    tail=tail.replace('_reuse_original_experts(x,wts,lids,state)',"rv_kernel_fn(dict(hidden_states=x,topk_ids=lids,topk_weights=wts,config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096)),state['moe_bank'])")
    block='"""Decode-layer MoE drop-in: moe_reuse_prepare(full_weights), moe_reuse_experts(x,wts,local_ids,state).\nRaw E4M3/NVFP4 banks only; FP32 gate and paired cold graph selection in prepare.\n"""\n'+native+'\n'+extra+'\n'+fixture+'\n'+tail
    component=ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/dropin.py';component.write_text(block)
    src=base.read_text();a=src.index('# MoE V9: persistent');b=src.index('\nCUDA_SRC = r"""',a)
    # Replace the old native bank definition, keeping the layer's attention/projection bodies frozen.
    src=src[:a]+src[b:]
    adapter="""
# The old layer prepare calls v9_prepare dynamically; the new component owns its entire packed bank.
def v9_prepare(weights):
    return moe_reuse_prepare(weights)


def experts_v9(x,wts,lids,state):
    return moe_reuse_experts(x,wts,lids,state['moe_bank'])
"""
    out=ROOT/'problems/muse_decode_layer/cand_r3_v9_reuse.py'
    out.write_text(src+'\n'+block+'\n'+adapter)
    for p in (component,out):print(p.relative_to(ROOT),hashlib.sha256(p.read_bytes()).hexdigest())


if __name__=='__main__':
    build_lab();build_candidate()
