import torch,deep_gemm
deep_gemm.use_deterministic_algorithms(True)
print("native mode", deep_gemm.get_deterministic_algorithms(),flush=True)
for M in (257,753,3012):
 a=torch.randn(M,8192,device='cuda',dtype=torch.bfloat16)
 b=torch.randn(2304,8192,device='cuda',dtype=torch.bfloat16)
 d=torch.empty(M,2304,device='cuda',dtype=torch.bfloat16)
 deep_gemm.bf16_gemm_nt(a,b,d)
 torch.cuda.synchronize()
 ref=torch.mm(a,b.T)
 print(M,((d.float()-ref.float()).norm()/ref.float().norm()).item(),flush=True)
