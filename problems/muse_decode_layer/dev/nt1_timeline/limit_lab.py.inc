_LIMIT_MODULE=[]
def limit_module():
 if not _LIMIT_MODULE:
  target='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
  cpp='''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
int64_t limit_init();int64_t limit_smem(int64_t);
void limit_loads(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
void limit_stream(std::vector<int64_t>,int64_t,int64_t,int64_t);
'''
  mod=load_inline(name='nt1_limit4_'+hashlib.sha256(LIMIT_C_CUDA.encode()).hexdigest()[:12],cpp_sources=cpp,cuda_sources=LIMIT_C_CUDA,
    functions=['limit_init','limit_smem','limit_loads','limit_stream'],no_implicit_headers=True,with_pytorch_error_handling=False,
    extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+target+',code=sm_'+target],verbose=False)
  mod.optin=mod.limit_init();_LIMIT_MODULE.append(mod)
 return _LIMIT_MODULE[0]

def limit_call(d,state,kind,experts):
 if kind=='A':return smallm_experts(d,state,69)
 if kind=='B':return trivial_experts(d,state,69)
 mod=limit_module();dev=d['hidden_states'].device;m=d['hidden_states'].shape[0];bank=state['banks']['lop3'];stream=torch.cuda.current_stream(dev).cuda_stream
 if kind=='C':
  base=smallm_module();smem=base.smem_needed(1,2,m);grid=min(264,state['sms']*base.resident_blocks(1,2,m))
  assert mod.limit_smem(2)<=smem<=mod.optin
  out=torch.empty(grid*256,device=dev,dtype=torch.int32)
  mod.limit_loads([d['topk_ids'].data_ptr(),bank['w13p'].data_ptr(),bank['w2p'].data_ptr(),out.data_ptr()],m,0,2,grid,smem,stream)
 else:
  grid=state['sms']*int(kind[1:]);out=torch.empty(grid*256,device=dev,dtype=torch.int32)
  mod.limit_stream([bank['w13p'].data_ptr(),bank['w2p'].data_ptr(),experts.data_ptr(),out.data_ptr()],experts.numel(),grid,stream)
 return out

def limit_profile(g,kind,reps=3):
 act=activity_module();act.profile_begin();ev=[]
 try:
  for _ in range(reps):
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();g.replay();b.record();b.synchronize();ev.append(a.elapsed_time(b)*1000)
 finally:
  torch.cuda.synchronize();rs=act.profile_end()
 name='moe_var' if kind in ('A','B') else 'loads_pipeline' if kind=='C' else 'stream_max'
 rs=[r for r in rs if name in r[0] and r[1][4]>0][-reps:]
 if len(rs)!=reps:raise ValueError(f'CUPTI {kind}: {len(rs)} records, need {reps}')
 return statistics.median((r[1][1]-r[1][0])/1000 for r in rs),statistics.median(ev),rs[-1][1][7:13]

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;device=next(v.device for v in weights.values() if isinstance(v,torch.Tensor))
 lines=['META P3a structural A/B/C/D. A=exact S11 acda6c62 flags69; B=trivial bounded BF16 reinterpret, WRONG MATH timing-only, same TMA/shared packed+scale loads, activation loads, MMA count and syncs; C=weight loads only, same tile addresses/order/grid/85KiB reservation/two-stage TMA depth, no expert-ready/finalizer or activation/MMA work; D4/D8/D16=pure streams of identical unique expert byte ranges, zero shared, 4/8/16 queued CTAs/SM, 256 threads. C/D outputs are checksums.',
  'META 16 public weight/router seeds, M8/16/32. Router produces fixed inputs outside each measured graph; each timed graph is ONE MoE/control node. Every replay starts with judge-style read flush; AB/BA x6 unprofiled, independent A:A; separately three CUPTI samples/arm/seed with observer event/off-event ratios. TB/s=unique packed expert bytes/node time; also report requested block-repeated bytes, not measured HBM transactions.']
 with ThreadPoolExecutor(max_workers=4) as pool:
  for j in [pool.submit(f) for f in (smallm_module,trivial_module,limit_module,activity_module)]:j.result()
 lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
 kinds=('A','B','C','D4','D8','D16');rows={};ratios={};bw={};observer={};noise={m:[] for m in (8,16,32)};fail=[];gate=0.;seedn=0;peak=0.
 oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
 try:
  for seed in range(16):
   if deadline-time.monotonic()<max(8,peak*1.2+2):lines.append(f'BUDGET before_seed={seed}');break
   started=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=device);w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
   st={k:base[k] for k in ('pnw','gs13','gs2','sms')};st['banks']={'lop3':base['bank']};smallm_bind(st,{8:69,16:69,32:69});trivial_bind(st,{8:69,16:69,32:69})
   for m in (8,16,32):
    pr,rs=routed_fixture(w,m,9000+m+97*seed);x,ids,tw,lids=route_call(pr,rs);torch.cuda.synchronize()
    d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
    counts=torch.bincount(lids[lids>=0].long(),minlength=32);experts=(counts>0).nonzero().flatten().int().contiguous();nb=(counts+7)//8
    unique_bytes=experts.numel()*28311552;requested_bytes=int(nb.sum())*28311552
    graphs={k:TimelineGraph(lambda k=k:limit_call(d,st,k,experts)) for k in kinds}
    want=nt1_fp32(d,w);ok,e,row=task_check(graphs['A'].replay(),want);assert ok,(m,seed,e,row);gate=max(gate,e)
    # All packed input words must contribute. Expected sum independently uses PyTorch int64 reductions.
    sums=[]
    for expert in range(32):
     if int(nb[expert]):
      total=sum(int(st['banks']['lop3'][b][expert].view(torch.int32).sum()) for b in ('w13p','w2p'));sums.append((total,int(nb[expert])))
    expect_unique=sum(t for t,n in sums)&0xffffffff;expect_repeat=sum(t*n for t,n in sums)&0xffffffff
    for k in ('C','D4','D8','D16'):
     out=graphs[k].replay();torch.cuda.synchronize();got=int(out.sum())&0xffffffff;expect=expect_repeat if k=='C' else expect_unique
     assert got==expect,(m,seed,k,got,expect)
    aa=TimelineGraph(lambda:limit_call(d,st,'A',experts));a,b,_=timeline_timing(graphs['A'],aa);noise[m].append(b-a)
    event={'A':a};order=list(kinds[1:]);order=order if seed%2==0 else order[::-1]
    for k in order:
     a,b,r=timeline_timing(graphs['A'],graphs[k]);event[k]=b;ratios.setdefault((m,k),[]).append(r)
    for k in kinds:
     node,evt,geom=limit_profile(graphs[k],k);rows.setdefault((m,k),[]).append(node);bw.setdefault((m,k),[]).append(unique_bytes/node/1e6);observer.setdefault((m,k),[]).append(evt/event[k])
     if seed==0:lines.append(f'NODE M{m} {k} node_us={node:.2f} event_us={event[k]:.2f} unique_MB={unique_bytes/1e6:.2f} requested_MB={requested_bytes/1e6:.2f} TBs={unique_bytes/node/1e6:.3f} gridX/Y/Z,threads,smem,regs={geom}')
    graphs={}
   seedn+=1;peak=max(peak,time.monotonic()-started)
 finally:torch.backends.cuda.matmul.allow_tf32=oldtf
 for m in (8,16,32):
  for k in kinds:
   v=rows.get((m,k),[])
   if not v:continue
   r=ratios.get((m,k),[1.]*len(v));se=statistics.stdev(r)/len(r)**.5 if len(r)>1 else 0.
   lines.append(f'AGG M{m} {k} node_us={statistics.mean(v):.3f} TBs={statistics.mean(bw[m,k]):.3f} paired_r={statistics.mean(r):.5f} se={se:.5f} observer={statistics.mean(observer[m,k]):.4f} n={len(v)}')
  if noise[m]:lines.append(f'AA M{m} d_us={statistics.mean(noise[m]):+.3f} abs={statistics.mean(map(abs,noise[m])):.3f}')
 lines.append(f'END seeds={seedn} A_worst_task_L2={gate:.5f} checksums=C+D_PASS seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}; B/C/D are diagnostics, no promotion')
 return lines
