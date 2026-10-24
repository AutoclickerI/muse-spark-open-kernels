"""Force Hopper-only prefill recipe selection/ABI with portable CUDA math.

This validates glue, capture and rejection. Mock ordering is a selection fixture,
never H200 timing or evidence that a CUTLASS recipe wins.
"""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time
import sys
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'problems/muse_bf16_proj'))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

p=load('prefill_cutlass','problems/muse_prefill_layer/cand_mid_cutlass.py')
lab=load('prefill_cutlass_lab','prof/lab/prefill_dense_cutlass.py')
ref=load('proj_reference','problems/muse_bf16_proj/reference.py')

class Abi:
    def __init__(self):
        self.handles={};self.tensors={};self.next=1;self.seen=set();self.calls=0;self.inject=None
    def vgc_error(self):return b'injected CUTLASS runtime/attribute failure'
    def vgc_create(self,tile,fp32,m,n,k,split,mode,sms,raster,swizzle,atomic):
        assert 0<=tile<8 and fp32 in (0,1) and mode in (0,1,2,3)
        assert split>=1 and (split==1 or mode in (0,2)) and sms>0
        assert raster in (0,1,2) and swizzle in (1,2,4,8) and atomic in (0,1)
        self.seen.add((m,n,k,fp32,tile,mode,split,sms,raster,swizzle,atomic))
        if self.inject=='create':self.inject=None;return None
        handle=self.next;self.next+=1;self.handles[handle]=(m,n,k,fp32)
        return handle
    def vgc_workspace(self,handle):assert handle in self.handles;return 256
    def vgc_info(self,handle,index):assert handle in self.handles;return (3,384,200000)[index]
    def vgc_destroy(self,handle):self.handles.pop(handle)
    def vgc_run(self,handle,x,w,y,ws,stream):
        m,n,k,fp32=self.handles[handle]
        a,b,c=(self.tensors[v] for v in (x,w,y))
        assert a.shape==(m,k) and b.shape==(n,k) and c.shape==(m,n)
        assert c.dtype==(torch.float32 if fp32 else torch.bfloat16)
        assert stream==torch.cuda.current_stream().cuda_stream and ws
        if self.inject=='attribute':self.inject=None;raise AttributeError('injected missing attribute')
        if self.inject=='runtime':self.inject=None;return -1
        torch.mm(a,b.T,out=c,**({'out_dtype':torch.float32} if fp32 else {}))
        if self.inject=='wrong':c.zero_()
        self.calls+=1
        return 0

abi=Abi()

class Function:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*args):return self.fn(*args)

def portable_module(module):
    original=module.VgcPlan
    class Plan(original):
        def run(self,x,w,out):
            abi.tensors={t.data_ptr():t for t in (x,w,out)}
            return super().run(x,w,out)
    module.VgcPlan=Plan
    functions=SimpleNamespace(**{name:Function(getattr(abi,name)) for name in
        ('vgc_error','vgc_create','vgc_workspace','vgc_info','vgc_destroy','vgc_run')})
    def compile_check(**kw):
        assert kw['cuda_sources']==module.VGC_SOURCE
        assert '-DNDEBUG' in kw['extra_cuda_cflags']
        assert module._vgos.environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
        assert kw['extra_include_paths']==[
            '/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include']
        return '/standin/prefill.so'
    with patch('torch.cuda.get_device_capability',return_value=(9,0)), \
         patch('torch.utils.cpp_extension.load_inline',side_effect=compile_check), \
         patch.object(module._vgct,'CDLL',return_value=functions):
        native=module.vgc_module()
    assert len(functions.vgc_create.argtypes)==11 and len(functions.vgc_run.argtypes)==6
    return native

pair_counts=[]
def paired_fixture(old,new,x,banks,flush,repeats=20,deadline=float('inf')):
    assert repeats in (4,20) and len(banks)==3
    if time.monotonic()>=deadline:raise TimeoutError('fixture budget')
    for fresh in banks:
        x.copy_(fresh);old.graph.replay();new.graph.replay()
        assert torch.isfinite(new.out).all()
    pair_counts.append(repeats)
    return 10.,9.,.9

def main():
    torch.manual_seed(47918)
    native=portable_module(p)
    lab_native=portable_module(lab)
    assert p.VGC_SOURCE==lab.VGC_SOURCE
    weights=ref.generate_weights(ref.WEIGHTS_SEED)
    hi=weights['router'].bfloat16();lo=(weights['router']-hi.float()).bfloat16()
    st={'qkv':weights['qkv_gate'],'o':weights['o'],'pooler':weights['pooler'],
        'unpooler':weights['unpooler'],'router_cat':torch.cat((hi,lo),0).contiguous(),'dense_choices':{}}
    p.pfc_prepare(st,time.monotonic()+240)
    assert not st['_pfc_plans'] and 'no H200-positive' in st['_pfc_notes'][0]
    # Enumerate every scheduling recipe at every exact shape through VgcPlan.
    enumeration=0
    for m in (753,3012,16384):
        for key,(n,k) in p.PFC_SHAPES.items():
            for recipe in p.pfc_choices(m,n,k,132):
                plan=native.plan(m,n,k,torch.float32 if key=='router_cat' else torch.bfloat16,recipe,'cuda')
                plan.close();enumeration+=1
    assert not abi.handles
    recipes={(m,key):[(0,1,1,132,0,1,0),(1,3,1,132,2,4,0),(7,2,4,132,1,1,1)]
             for m in (753,3012,16384) for key in p.PFC_SHAPES}
    with patch.object(p,'vgc_module',return_value=native),patch.object(p,'pfc_pair',side_effect=paired_fixture):
        p.pfc_prepare(st,time.monotonic()+270,recipes)
    assert len(st['_pfc_plans'])==15,st['_pfc_notes']
    replays=0
    for (key,m,recipe),plan in st['_pfc_plans'].items():
        w=st[key];x=torch.randn(m,w.shape[1],device='cuda',dtype=torch.bfloat16)
        with p._PgGraph(lambda:p._score_dense(x,st,key)) as graph:
            for _ in range(2):
                x.normal_();graph.graph.replay()
                p.pfc_check(graph.out,x.float()@w.float().T,key=='router_cat');replays+=1
        for bad_x in (x[:,::2],x.float()):
            try:plan.run(bad_x,w,torch.empty(m,w.shape[0],device='cuda',dtype=plan.dtype))
            except ValueError:pass
            else:raise AssertionError('shape/dtype guard did not reject')
    # Every optional failure keeps the old choice and releases native handles.
    one={(753,'qkv'):[(0,3,1,132,0,1,0)]}
    for inject in ('create','attribute','runtime','wrong'):
        fresh={**st,'dense_choices':{}}
        abi.inject=inject
        with patch.object(p,'vgc_module',return_value=native),patch.object(p,'pfc_pair',side_effect=paired_fixture):
            p.pfc_prepare(fresh,time.monotonic()+240,one)
        abi.inject=None
        assert not fresh['_pfc_plans'] and not fresh['dense_choices'],(inject,fresh['_pfc_notes'])
        assert any('REJECT' in row for row in fresh['_pfc_notes'])
    fresh={**st,'dense_choices':{}}
    p.pfc_prepare(fresh,time.monotonic()-1,one)
    assert not fresh['_pfc_plans'] and any('BUDGET' in row for row in fresh['_pfc_notes'])
    # Force the lab's exact Lt32 control ABI and DeepGEMM glue with torch math.
    class Lt:
        def lt_options(self,w,x,out):return torch.empty(32,64,dtype=torch.uint8)
        def lt_run(self,w,x,out,workspace,codes,index):
            assert 0<=index<len(codes) and workspace.numel()==256<<20
            torch.mm(x,w.T,out=out,**({'out_dtype':torch.float32} if out.dtype==torch.float32 else {}))
    import deep_gemm
    def dg(x,w,out):torch.mm(x,w.T,out=out)
    def subset(m,n,k,sms):
        for tile in range(8):yield (tile,1,1,132,0,1,0)
        yield (1,3,1,132,0,1,0)
        yield (7,2,4,132,2,4,1)
        yield (0,0,1,132,0,1,0)
    with patch.object(lab,'vgc_module',return_value=lab_native), \
         patch.object(lab,'pfc_lt_module',return_value=Lt()), \
         patch.object(lab,'pfc_choices',side_effect=subset), \
         patch.object(lab,'pfc_pair',side_effect=paired_fixture), \
         patch.object(deep_gemm,'bf16_gemm_nt',side_effect=dg):
        rows=lab.lab(time.monotonic()+285,weights)
    measurements=[json.loads(row.split(' ',1)[1]) for row in rows if row.startswith('PFC_ROW ')]
    assert len(measurements)==15,rows
    assert all(row['tried']==11 and row['rejected']==0 for row in measurements),rows
    assert sum(map(lambda row:len(row)+1,rows))<9000
    assert len([r for r in pair_counts if r==20])>=75
    for plan in st['_pfc_plans'].values():plan.close()
    assert not abi.handles and p._PgGraph.active==0 and lab._PgGraph.active==0
    report={'ok':True,'sha256':hashlib.sha256((ROOT/'problems/muse_prefill_layer/cand_mid_cutlass.py').read_bytes()).hexdigest(),
            'recipe_ABI_cases':enumeration,'portable_ABI_calls':abi.calls,'fresh_selected_graph_replays':replays,
            'forced_lab_shapes':len(measurements),'lab_log_chars':sum(len(r)+1 for r in rows),
            'failure_gates':['compile','create','attribute','runtime','FP32_accuracy','budget','shape','dtype'],
            'timings':'mock selection fixtures only; no Hopper execution or speed claim'}
    # Module compile failure must reject independently.
    fresh={**st,'dense_choices':{}}
    with patch.object(p,'vgc_module',side_effect=RuntimeError('injected compile failure')):
        p.pfc_prepare(fresh,time.monotonic()+240,one)
    assert not fresh['_pfc_plans'] and any('REJECT' in r for r in fresh['_pfc_notes'])
    (ROOT/'runs/screen/prefill_cutlass_glue.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
