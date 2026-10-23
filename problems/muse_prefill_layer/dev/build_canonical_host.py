from pathlib import Path
import re
p=Path(__file__).resolve().parents[1]
h=(p/'dev/moe_host.py').read_text()
a=h.index('def _blob_index');b=h.index('_mod = None',a)
h=h[:a]+h[b:]
h=h.replace('name="muse_prefill_grouped_w4a16"','name="muse_prefill_canonical_pipeline"')
h=h.replace('torch::Tensor sfblob, torch::Tensor gs','torch::Tensor gs, torch::Tensor gs_up')
# Gather natural x into V6-permuted K positions; read outputs through same inverse permutation.
h=h.replace('    cc = c % 64\n', '    cc0 = c % 64\n    pair = cc0 // 2\n    cc = 2*((pair//8)+((pair&1)*4)+((pair&6)*4))+(cc0&1)\n')
# The combine has an extra indentation level.
h=h.replace('            cc = c % 64\n', '            cc0 = c % 64\n            pair = cc0 // 2\n            cc = 2*((pair//8)+((pair&1)*4)+((pair&6)*4))+(cc0&1)\n')
# Fix generic replacement's indentation inside combine.
h=h.replace('            cc0 = c % 64\n    pair = cc0 // 2\n    cc =', '            cc0 = c % 64\n            pair = cc0 // 2\n            cc =')
a=h.index('def experts_tiled(')
h=h[:a]+'''def run_raw(xn, topk_w, slot_e, slot_pos, counts, st):
    T, D = xn.shape
    dev = xn.device
    max_tiles = triton.cdiv(T * 8 + 32 * (BN - 1), BN)
    tstart = torch.empty(33, dtype=torch.int32, device=dev)
    tile_offsets_kernel[(1,)](counts, tstart, BNC=BN)
    xt = torch.empty((max_tiles * BN * D,), dtype=torch.bfloat16, device=dev)
    rows = torch.empty((T, 8), dtype=torch.int32, device=dev)
    gather_tiled_kernel[(T,)](xn, slot_e, slot_pos, tstart, xt, rows, D=D, BNC=BN, num_warps=4)
    inter = torch.empty_like(xt)
    y = torch.empty_like(xt)
    mod=st['tiled_mod']
    mod.moe_gemm(st['w13p'], st['gs13_gate'], st['gs13_up'], xt, inter, tstart, 32,64,64,1,st['sms'])
    mod.moe_gemm(st['w2p'], st['gs2'], st['gs2'], inter, y, tstart, 16,64,64,0,st['sms'])
    out = torch.empty_like(xn)
    combine_tiled_kernel[(T,)](y, rows, topk_w, st['post_expert'], out, D=D, BNC=BN,num_warps=4)
    return out


def experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):
    xn=torch.empty_like(pooled)
    _pre_expert_norm_kernel[(pooled.shape[0],)](pooled,st['pre_expert'],xn,D=D,num_warps=4)
    return run_raw(xn,topk_w,slot_e,slot_pos,counts,st)
'''
# Lossless canonical repack from the MoE owner's contract implementation (read-only input).
r=(p.parent/'muse_moe_w4a16/cand_rs128.py').read_text()
a=r.index('_DQ_SLOTS =');b=r.index('\ndef _prepare_native(',a)
h+='\n'+r[a:b]+'''

def repack_v6(wq,sf,gated):
    E,R,_=wq.shape
    dev=wq.device
    if gated:
        t=torch.arange(R//16,device=dev)[:,None]
        i=torch.arange(8,device=dev)[None,:]
        rows=torch.cat((t*8+i,R//2+t*8+i),1)
    else:
        rows=torch.arange(R,device=dev).view(-1,16)
    return torch.stack([_repack(wq[e],sf[e],rows) for e in range(E)])
'''
(p/'dev/moe_canonical_host.py').write_text(h)
# Standalone full-layer candidate. None of the dev imports enter the candidate.
v=(p/'cand_v3.py').read_text()
a=v.index('    try:\n        st["bank"] = _marlin_bank(weights)');b=v.index('    torch.cuda.synchronize()',a)
v=v[:a]+'    st["bank"] = None\n'+v[b:]
a=v.index('CUDA_SRC = ')
# All trailing helper code is replaced with canonical helper, keeping layer functions above it.
front=v[:a]
i=front.index('    if T >= BIG_TOKENS or st["bank"] is None:')
j=front.index('    if T <= 8:',i)
front=front[:i]+'    moe = experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st)\n'+front[j:]
for name in ('_marlin_bank','_bench_w4a16','_experts_marlin','_experts_dequant'):
    front=re.sub(r'\ndef '+name+r'\(.*?(?=\ndef |\Z)', '\n', front, flags=re.S)
v=front+"CUDA_SRC = "+repr((p/'dev/moe_w4a16_canonical.cu').read_text())+'\n\n'+h.replace('src = open(src_path).read()','src = CUDA_SRC')
v+='''

def prepare(weights):
    st=prepare_base(weights)
    st['w13p']=repack_v6(weights['w13'],weights['w13_sf'],True)
    st['w2p']=repack_v6(weights['w2'],weights['w2_sf'],False)
    st['gs13_gate']=weights.get('gs13_gate',weights['w13_gs']).float().contiguous()
    st['gs13_up']=weights.get('gs13_up',weights['w13_gs']).float().contiguous()
    st['gs2']=weights['w2_gs'].float().contiguous()
    st['post_expert']=st['post_expert_bf16']
    st['tiled_mod']=load_module(None)
    st['sms']=torch.cuda.get_device_properties(weights['qkv_gate'].device).multi_processor_count
    for key in ('w13','w13_sf','w2','w2_sf'):
        st.pop(key,None)
    return st
'''
(p/'cand_canonical_prefill.py').write_text(v)
print('built',p/'cand_canonical_prefill.py')
