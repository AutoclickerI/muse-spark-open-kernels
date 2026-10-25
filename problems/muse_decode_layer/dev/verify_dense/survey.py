from pathlib import Path
import hashlib,json,time
import torch,deep_gemm,cutlass
HERE=Path(__file__).resolve().parent
r=dict(deep_gemm_version=deep_gemm.__version__,cutlass_dsl_version=getattr(cutlass,'__version__','unknown'),deep_gemm_bf16={},sources={})
for path in (Path(deep_gemm.__file__),Path(deep_gemm.__file__).parent/'include/deep_gemm/impls/sm90_bf16_gemm.cuh',Path(cutlass.__file__).parent/'utils/hopper_helpers.py'):
 r['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
for dtype in (torch.bfloat16,torch.float32):
 x=torch.randn((64,8192),device='cuda',dtype=torch.bfloat16);w=torch.randn((2304,8192),device='cuda',dtype=torch.bfloat16)
 out=torch.empty((64,2304),device='cuda',dtype=dtype)
 try:
  started=time.monotonic();deep_gemm.bf16_gemm_nt(x,w,out,compiled_dims='mnk');torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):deep_gemm.bf16_gemm_nt(x,w,out,compiled_dims='mnk')
  x.normal_();graph.replay();ref=x.float()@w.float().T
  err=float((out.float()-ref).norm()/ref.norm());assert err<(1e-4 if dtype==torch.float32 else .01),err
  r['deep_gemm_bf16'][str(dtype)]=dict(available=True,error=err,compile_s=time.monotonic()-started,graph=True)
 except Exception as e:r['deep_gemm_bf16'][str(dtype)]=dict(available=False,error=str(e)[:1000])
(HERE/'survey.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
