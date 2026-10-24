"""Pack the W2-tail repair and matched Humming/pipeline controls into one deadline-bounded lab."""
import ast,hashlib
from pathlib import Path
from build_prefill_lab import subset
from native_abi import pipeline,CPP_PRELUDE,PIPE_CPP
H=Path(__file__).resolve().parent;R=H.parents[3]

CONTROL='''
def wx_control_prepare(w):
    wx_need(6.,'humming compile')
    dev=w['w13'].device
    hum=pf__Experts(dict(w,w13_gs=torch.ones(32,device=dev)),0,32)
    hum.gemm_type=pf_GemmType.GROUPED_MASKED
    wx_need(6.,'humming compile256');hum.compile([256])
    for key in ('w13','w2'):assert hum.weight(key)['weight_scale'].dtype==torch.float8_e4m3fn
    return dict(pre_expert=torch.ones(4096,device=dev),ones=torch.ones(32,device=dev),
        post_expert=(1+w['post_norm_w'].float()).bfloat16(),humming=hum,
        gs13_gate=w['w13_gs_gate'],gs13_up=w['w13_gs_up'],gs2=w['w2_gs'])
'''

def inline(filename,var,cuda):
 tree=ast.parse((H/filename).read_text());tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='engine')]
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==var for x in n.targets):n.value=ast.Constant(cuda)
  if isinstance(n,ast.FunctionDef) and n.name in ('pipe_module','fx_module'):
   n.decorator_list=[];n.body=ast.parse('return wx_module()').body
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='PIPE_CPP' for a in n.targets):n.value=ast.Constant(PIPE_CPP)
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='pipe_run':
   tensors=n.args[:10];out=tensors[-1];rest=n.args[10:]
   ptrs=ast.parse('[v.data_ptr() for v in values]',mode='eval').body;ptrs.generators[0].iter=ast.List(elts=tensors,ctx=ast.Load())
   n.args=[ptrs]+rest+[ast.Call(func=ast.Attribute(value=out,attr='size',ctx=ast.Load()),args=[ast.Constant(-2)],keywords=[]),ast.Call(func=ast.Attribute(value=out,attr='size',ctx=ast.Load()),args=[ast.Constant(-1)],keywords=[]),ast.parse('torch.cuda.current_stream().cuda_stream',mode='eval').body]
 return ast.unparse(ast.fix_missing_locations(tree))

def main():
 original=(R/'prof/lab/moe_fused_m256.py').read_text();control=subset(original+'\n'+CONTROL,('dc_weights','dc_fixtures','dc_oracles','wx_control_prepare','pp_reference','MWGraph'))
 engine=subset((H/'engine.py').read_text(),('mw_repack_planes','mw_route','mw_prefix','mw_swiglu','mw_norm','mw_combine'))
 pipe=(H/'pipeline.cu').read_text();a=pipe.index('#define PIPE_CASES');b=pipe.index('\n}',a)
 pipe=pipe[:a]+'#define PIPE_CASES(F) \\\n F(1,16,128,4,1,true) F(5,32,128,4,1,true)'+pipe[b:]
 pipe=pipeline(pipe)
 fixed=(H/'w2fix.cu').read_text();module='''
WX_DEADLINE=float('inf')
class WxBudget(Exception):pass

def wx_need(seconds,where):
    left=WX_DEADLINE-time.monotonic()
    if left<seconds:raise WxBudget(where+' left='+format(left,'.1f'))

@functools.lru_cache(None)
def wx_module():
    wx_need(40.,'native compile')
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='w2fix_lab_'+hashlib.sha256(WX_CUDA.encode()).hexdigest()[:12],
        cpp_sources=FX_CPP+PIPE_CPP,cuda_sources=WX_CUDA,
        functions=['pipe_init','pipe_info','pipe_run','fx_init','fx_info','fx_route','fx_run','fx_finish'],
        no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.pipe_init();mod.fx_init();return mod
'''
 out=R/'prof/lab/moe_w2fix_m256.py';out.write_text(control+'\n'+engine+'\n'+inline('pipeline.py','PIPE_CUDA',pipe)+'\n'+inline('w2fix.py','FX_CUDA',fixed)+'\nWX_CUDA='+repr(pipe+'\n'+fixed)+'\n'+module+'\n'+(H/'w2fix_driver.py.inc').read_text())
 print(out.relative_to(R),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)

if __name__=='__main__':main()
