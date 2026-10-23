import torch, time
import deep_gemm as dg
print("align", dg.get_mk_alignment_for_contiguous_layout(), dg.get_m_alignment_for_contiguous_layout() if hasattr(dg,'get_m_alignment_for_contiguous_layout') else None)
E, N, K = 4, 1024, 512
counts = [100, 0, 300, 37]
align = dg.get_mk_alignment_for_contiguous_layout()
starts=[]; s=0; ends=[]
for c in counts:
    starts.append(s); ends.append(s+c); s = ((s+c+align-1)//align)*align
M = s + align*2
a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
b = torch.randn(E, N, K, device='cuda', dtype=torch.bfloat16)
d = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
psum = torch.tensor(ends, dtype=torch.int32, device='cuda')
t0=time.time()
try:
    dg.m_grouped_bf16_gemm_nt_contiguous(a, b, d, psum, use_psum_layout=True)
    torch.cuda.synchronize(); print("psum ok", time.time()-t0)
    err=0
    for e,(st,c) in enumerate(zip(starts,counts)):
        if c==0: continue
        ref = a[st:st+c].float() @ b[e].float().T
        err=max(err, ((d[st:st+c].float()-ref).norm()/ref.norm()).item())
    print("err", err)
except Exception as ex: print("psum fail", repr(ex))
# m_indices layout
mi = torch.full((M,), -1, dtype=torch.int32, device='cuda')
for e,(st,c) in enumerate(zip(starts,counts)): mi[st:st+c]=e
d2 = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
try:
    dg.m_grouped_bf16_gemm_nt_contiguous(a, b, d2, mi)
    torch.cuda.synchronize(); print("mindices ok")
except Exception as ex: print("mi fail", repr(ex))
