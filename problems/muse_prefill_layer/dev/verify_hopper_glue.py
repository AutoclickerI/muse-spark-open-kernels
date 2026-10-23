"""Force Hopper-only Python dispatch with portable native GEMM and tensor/tuple FA3 stand-ins."""
import sys,types,torch,importlib.util
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import local_eval as le
import reference as ref
path=Path(sys.argv[1]);m=le.load_module('hopper_glue_candidate',path)
original=sys.modules.get('sgl_kernel.flash_attn');stub=types.ModuleType('sgl_kernel.flash_attn');sys.modules['sgl_kernel.flash_attn']=stub
m.IS_HOPPER=True
try:
 for window in ('full','sliding'):
  data=ref.generate_input(batch=1,ctx=1025,layer=window,q_len=139,page=64,kv_scale=1/64,seed=52991)
  q=torch.randn(139,16*64,device='cuda').to(torch.float8_e4m3fn)
  for tuple_out in (False,True):
   def fake(**kw):
    out=torch.zeros_like(kw['q'],dtype=torch.bfloat16)
    return (out,torch.empty(0,device=q.device)) if tuple_out else out
   stub.flash_attn_with_kvcache=fake
   for mode in ('paged','contiguous','unpacked'):
    m._ATTENTION_MODES={f'{load}:{int(window=="sliding")}':mode for load in ('turn 1','turn 4','first turn')}
    for _ in range(2):m._attention_selected(data,q,{})
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=m._attention_selected(data,q,{})
    g.replay();torch.cuda.synchronize()
    assert out.shape==(139,1024) and out.dtype==torch.bfloat16
    del g,out
  print(window,'FA3 tensor/tuple, all selected paths graph-capture PASS',flush=True)
 import deep_gemm
 deep_gemm.use_deterministic_algorithms(True)
 m._DENSE_MODES={x:'deepgemm' for x in ('turn 1','turn 4','first turn')}
 a=torch.randn(153,256,device='cuda',dtype=torch.bfloat16);w=torch.randn(128,256,device='cuda',dtype=torch.bfloat16)
 for _ in range(2):m._dense_selected(a,w)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=m._dense_selected(a,w)
 a.normal_();g.replay();torch.cuda.synchronize();expected=a.float()@w.float().T
 err=float((out.float()-expected).norm()/expected.norm());assert err<0.004
 print('Forced native dense path fresh replay PASS, relative L2',err,flush=True)
finally:
 if original is None:sys.modules.pop('sgl_kernel.flash_attn',None)
 else:sys.modules['sgl_kernel.flash_attn']=original
