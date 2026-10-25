"""Compile the CUDA13 DAG query and verify captured fork/join connectivity locally."""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
src=(REPO/'problems/muse_decode_layer/dev/timeline/activity.cpp').read_text()+'\n'+(HERE/'dag.cpp.inc').read_text()
act=load_inline(name='bd_dag_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources=src,
    functions=['profile_begin','profile_end','capture_nodes','capture_dag','exec_id'],
    no_implicit_headers=True,with_pytorch_error_handling=False,extra_cflags=['-O2','-std=c++20'],
    extra_include_paths=['/usr/local/cuda/include'],extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-lcudart'],verbose=False)
x=torch.ones(1024,device='cuda');y=torch.ones_like(x);side=torch.cuda.Stream()
x.add_(1);y.mul_(2);torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    x.add_(1);side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):y.mul_(2)
    torch.cuda.current_stream().wait_stream(side);x.add_(y)
    rows,edges=act.capture_dag(torch.cuda.current_stream().cuda_stream)
assert sum(r[1]==0 for r in rows)==3,(rows,edges)
assert len(edges)>=2 and all(0<=e[0]<len(rows) and 0<=e[1]<len(rows) for e in edges)
act.profile_begin()
for _ in range(5):g.replay()
torch.cuda.synchronize();records=act.profile_end()
eid=act.exec_id(g.raw_cuda_graph_exec());ids={(eid<<32)|(r[0]&0xffffffff) for r in rows if r[1]==0}
assert {r[4] for _,r in records if r[4]}==ids
(HERE/'dag_validation.json').write_text(json.dumps(dict(nodes=rows,edges=edges,profiled_records=len(records),replays=5,
    cupti_rekey=True,source_sha256=hashlib.sha256(src.encode()).hexdigest()),indent=2)+'\n')
print('PASS DAG fork/join',len(rows),'nodes',len(edges),'edges and CUPTI rekey')

# Driver-launched Triton kernels and programmatic edges must be queryable too.
import triton
import triton.language as tl
@triton.jit
def pdl_probe(X):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    i=tl.arange(0,128);tl.store(X+i,tl.load(X+i)+1)
pdl_probe[(1,)](x,launch_pdl=True);torch.cuda.synchronize()
g2=torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    pdl_probe[(1,)](x,launch_pdl=True);pdl_probe[(1,)](x,launch_pdl=True)
    rows2,edges2=act.capture_dag(torch.cuda.current_stream().cuda_stream)
assert len(rows2)==2 and any(e[2]==1 for e in edges2),(rows2,edges2)
g2.replay();torch.cuda.synchronize()
p=HERE/'dag_validation.json';data=json.loads(p.read_text());data['programmatic_nodes']=rows2;data['programmatic_edges']=edges2;data['triton_driver_nodes']=True;p.write_text(json.dumps(data,indent=2)+'\n')
print('PASS Triton driver graph nodes and programmatic edge type/ports')
