"""Profile a private copy of the frozen K128 source with per-CTA clock64 marks."""
import json
import statistics
import sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root.parents[1] / 'tools')]
import local_eval as E

m = E.load_module('k128_profile_base', root / 'cand_xqa_k128.py')
ref = E.load_module('k128_profile_ref', root / 'reference.py')
ref._fa3_scheduler_metadata = lambda *a, **kw: None
c = m.X_CUDA
c = c.replace('float* partial; int* count;', 'uint64_t* trace; float* partial; int* count;')
c = c.replace('uint8_t* vh){\n if constexpr', 'uint8_t* vh,uint64_t* tr){\n if(threadIdx.x==0)tr[3]=clock64();\n if constexpr')
c = c.replace(' int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;\n float s',
              ' if(threadIdx.x==0)tr[4]=clock64();\n int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;\n float s')
qkend = ' #pragma unroll\n for(int nt=0;nt<1;nt++){\n  #pragma unroll\n  for(int e=0;e<2;e++)'
c = c.replace(qkend, ' if(threadIdx.x==0)tr[5]=clock64();\n' + qkend, 1)
c = c.replace(' if constexpr(P8){\n  #pragma unroll', ' if(threadIdx.x==0)tr[6]=clock64();\n if constexpr(P8){\n  #pragma unroll', 1)
c = c.replace('\n}\ntemplate<int ST,bool P8>\n__global__', '\n if(threadIdx.x==0)tr[7]=clock64();\n}\ntemplate<int ST,bool P8>\n__global__', 1)
c = c.replace(' extern __shared__ __align__(128) uint8_t sm[];', ''' uint64_t* trace=p.trace+((blockIdx.z*p.hkv+blockIdx.y)*p.ns+blockIdx.x)*1024*16;
 if(threadIdx.x==0){trace[6]=0;trace[0]=clock64();}
 extern __shared__ __align__(128) uint8_t sm[];''', 1)
c = c.replace(' load_rows(ST-1);', ' load_rows(ST-1);\n if(tid==0)trace[1]=clock64();', 1)
c = c.replace(' float scale=p.scale*(*p.ks);', ' if(tid==0)trace[2]=clock64();\n float scale=p.scale*(*p.ks);', 1)
c = c.replace('  wait<ST-2>();__syncthreads();', '  uint64_t* tr=trace+(i+1)*16;\n  if(tid==0)tr[0]=clock64();\n  wait<ST-2>();__syncthreads();\n  if(tid==0)tr[1]=clock64();', 1)
c = c.replace('  load_rows(i+ST);', '  load_rows(i+ST);\n  if(tid==0)tr[2]=clock64();', 1)
c = c.replace('p.window,scale,vh);\n  __syncthreads();', 'p.window,scale,vh,tr);\n  __syncthreads();\n  if(tid==0)tr[8]=clock64();', 1)
c = c.replace(' if(last)tile<32,P8>(st,nk,nv,seq,seq+8,seq,p.window,scale,vh);',
              ' if(tid==0)trace[3]=clock64();\n if(last)tile<32,P8>(st,nk,nv,seq,seq+8,seq,p.window,scale,vh,trace+1000*16);\n if(tid==0)trace[4]=clock64();', 1)
c = c.replace(' __syncthreads();if(!elected)return;', ' __syncthreads();if(tid==0)trace[5]=clock64();\n if(!elected)return;', 1)
c = c.replace(' __syncthreads();if(tid==0)atomicExch', ' __syncthreads();if(tid==0)trace[6]=clock64();\n if(tid==0)atomicExch', 1)
c = c.replace('int64_t stream){\n XP p{};', 'int64_t stream,int64_t trace){\n XP p{};p.trace=(uint64_t*)trace;', 1)
cpp = m.X_CPP.replace('int64_t stream);', 'int64_t stream,int64_t trace);')
(root / 'dev/k128_profile.cu').write_text(c)
arch = ''.join(map(str, torch.cuda.get_device_capability()))
ext = load_inline(name=f'muse_k128_profile_sm{arch}', cpp_sources=cpp, cuda_sources=c, functions=['xforward'],
    extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '-lineinfo', '-Xptxas=-v',
                      f'-gencode=arch=compute_{arch},code=sm_{arch}'], verbose=False)
orig = m._xext()
timer = E.Timer()
rows = []
for batch in (16, 32):
    data = ref.generate_input(batch=batch, ctx=80000, heads='tp8', layer='full', q_len=8,
                              spread='agentic', dist='coherent', page=1, seed=43)
    for p8, stages, waves in [(False, 3, w) for w in (1, 2, 4, 8)] + [(True, 3, 4)]:
        m._w8_set((p8, stages, waves))
        m._xext = lambda: orig
        graph, out = E.capture(m._xqa_verify, data)
        us = statistics.median(timer.replay_us(graph) - timer.fixed_us for _ in range(7))
        del graph, out
        ns = max(1, min(64, (86008+127)//128, (m._num_sms(data['qkv'].device)*waves+batch*2-1)//(batch*2)))
        trace = torch.zeros((batch*2*ns, 1024, 16), dtype=torch.int64, device='cuda')
        class Proxy:
            def xforward(self, *args):
                return ext.xforward(*args, trace.data_ptr())
        m._xext = lambda: Proxy()
        graph, out = E.capture(m._xqa_verify, data)
        trace.zero_()
        instrumented_us = statistics.median(timer.replay_us(graph) - timer.fixed_us for _ in range(3))
        tr = trace.cpu()
        valid = tr[:, 1:999, 0] != 0
        s = tr[:, 1:999, :][valid]
        parts = {name: float((s[:,b]-s[:,a]).double().mean()) for name,a,b in
                 [('wait',0,1),('issue',1,2),('widen',3,4),('qk',4,5),('softmax',5,6),('pv',6,7),('barrier',7,8)]}
        hdr = tr[:,0,:]
        elected = hdr[:,6] != 0
        prep = (hdr[:,2]-hdr[:,0]).double()
        merge = (hdr[elected,6]-hdr[elected,5]).double()
        row = dict(batch=batch,p8=p8,stages=stages,waves=waves,splits=ns,us=us,
                   instrumented_us=instrumented_us,tile_cycles=parts,
                   prep_cycles=float(prep.mean()),merge_cycles=float(merge.mean()),
                   tile_count=int(valid.sum()),elected=int(elected.sum()))
        rows.append(row)
        print(json.dumps(row), flush=True)
        (root / 'dev/k128_stage_profile_local.json').write_text(json.dumps(rows, indent=2))
        del graph, out, trace
    del data
    torch.cuda.empty_cache()
