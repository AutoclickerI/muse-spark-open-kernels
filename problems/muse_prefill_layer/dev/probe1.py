import torch, time
print(torch.__version__, torch.cuda.get_device_name())
# 1. torch._grouped_mm
E,K,N=4,512,256
offs=torch.tensor([100,250,250,400],dtype=torch.int32,device='cuda')
a=torch.randn(512,K,device='cuda',dtype=torch.bfloat16)
b=torch.randn(E,N,K,device='cuda',dtype=torch.bfloat16)
try:
    out=torch._grouped_mm(a,b.transpose(-2,-1),offs=offs)
    ref=torch.cat([a[0:100]@b[0].T,a[100:250]@b[1].T,a[250:400]@b[3].T])
    print("grouped_mm ok", out.shape, (out[:400].float()-ref.float()).abs().max().item())
except Exception as e: print("grouped_mm fail", e)
# 2. deep_gemm
try:
    import deep_gemm
    print("deep_gemm", deep_gemm.__file__, [x for x in dir(deep_gemm) if 'bf16' in x or 'align' in x])
except Exception as e: print("deep_gemm import fail", repr(e))
# 3. quack w4a16
try:
    from quack.gemm_w4 import gemm_w4a16, prepare_w4_weight
    print("quack ok")
except Exception as e: print("quack import fail", repr(e))
