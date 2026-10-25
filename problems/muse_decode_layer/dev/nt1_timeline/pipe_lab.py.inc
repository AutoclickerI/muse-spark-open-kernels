PIPE_MICRO_CPP=r'''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdint.h>
std::vector<int64_t> pipe_op_run(std::vector<int64_t>,int,int,int,int);
std::vector<int64_t> pipe_work_run(std::vector<int64_t>,int,int,int,int,int);
void pipe_lat_run(std::vector<int64_t>,int,int,int,int);
'''
_PIPE_MODULE=[]
def pipe_module():
 if not _PIPE_MODULE:
  target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
  m=load_inline(name='nt1_pipe_'+hashlib.sha256(PIPE_MICRO_CUDA.encode()).hexdigest()[:12],cpp_sources=PIPE_MICRO_CPP,cuda_sources=PIPE_MICRO_CUDA,
   functions=['pipe_op_run','pipe_work_run','pipe_lat_run'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
  _PIPE_MODULE.append(m)
 return _PIPE_MODULE[0]

def pipe_call(fn,ptr,*args):
 ptr[-1]=torch.cuda.current_stream().cuda_stream
 try:return fn(ptr,*args)
 except RuntimeError as e:raise RuntimeError(f'{fn.__name__} args={args}: {e}') from e

class PipeGraph:
 def __init__(self,fn):
  fn();torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph):self.geometry=fn()
 def replay(self):self.graph.replay()

def pipe_time(g,reps=3):
 vals=[]
 for _ in range(reps):
  lab_cold_l2();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();vals.append(a.elapsed_time(b)*1000)
 return statistics.median(vals)

def pipe_profile(g,tag='pipe_'):
 mod=activity_module();mod.profile_begin()
 try:
  for _ in range(3):lab_cold_l2();g.replay();torch.cuda.synchronize()
 finally:rows=mod.profile_end()
 times=[(v[1]-v[0])/1000 for k,v in rows if tag in k and v[4]>0][-3:]
 if len(times)!=3:raise RuntimeError('missing pipe CUPTI records')
 return statistics.median(times)

def pipe_routed(pr,rs,st,arm,trace=False,ts=None):
 x,ids,tw,lids=route_call(pr,rs);m=x.shape[0];d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 if arm=='S14':out=smallm_experts(d,st,69)
 elif arm=='PCLK':out=pclk_experts(d,st,325 if trace else 69,stamps=ts)
 else:out=pdia_experts(d,st,325 if trace else 69,stamps=ts)
 return out,x,ids,tw,lids

def pipe_ids(counts,grid):
 # Exact unit order (e,m-chunk,token-block,m), per-pass chunk sizes as S14, before CTA range slicing.
 rows=[]
 for ntile,offset in ((512,0),(256,16384)):
  blocks=[(n+7)//8 for n in counts];total=sum(blocks)*ntile;cc=min(256,max(1,(total+grid//2)//grid))
  seq=[]
  for e,nb in enumerate(blocks):
   for mb in range(0,ntile,cc):
    for b in range(nb):seq.extend(offset+e*ntile+m for m in range(mb,min(ntile,mb+cc)))
  rows.append(seq)
 # Put each CTA's W13 and W2 intervals together; pure copy has no expert-ready dependency.
 result=[]
 for c in range(grid):
  for seq in rows:result.extend(seq[len(seq)*c//grid:len(seq)*(c+1)//grid])
 return torch.tensor(result,device='cuda',dtype=torch.int32)

def pipe_op_table(mod,out,stamps,sms,deadline):
 lines=[];names=['LOP3','SHF','PRMT','HMUL2','FFMA','HMMA','LDS128','LDS32','LDS32_GATHER4','SHFL','MUFU']
 for op,name in list(enumerate(names))+[(13,'LDS_ADDR_CHAIN')]:
  for ilp in (1,4):
   for cap in (1,2,4,8):
    if deadline-time.monotonic()<6:raise TimeoutError('pipe table incomplete')
    ptr=[out.data_ptr(),stamps.data_ptr(),0,0,torch.cuda.current_stream().cuda_stream]
    g=PipeGraph(lambda:pipe_call(mod.pipe_op_run,ptr,op,ilp,cap,256));node=pipe_profile(g);g.replay();torch.cuda.synchronize();G=g.geometry[0];v=stamps[:G,:8].cpu().tolist()
    clk=statistics.median((r[3]-r[2])/(r[1]-r[0]) for r in v if r[1]>r[0]);span=(max(r[1] for r in v)-min(r[0] for r in v))/1000
    instr=G*8*256*8*ilp;rate=instr/(span*1000*clk*sms);active=len(set(r[6] for r in v))
    lines.append(f'PIPE {name} ILP{ilp} cap{cap} G/res/regs/local/shared={g.geometry} activeSM={active} node_us={node:.3f} clock_GHz={clk:.4f} op_warp_per_SM_cycle={rate:.5f} CTAcycles={statistics.median(r[3]-r[2] for r in v):.1f}')
 return lines

def compact_pipe(lines):
 import base64,lzma
 # Preserve the full diagnostic matrix under h200_lab's 9000-character receipt cap.
 # The packed body is ordinary text, decoded losslessly by read_pipe.py.
 detail='\n'.join(x for x in lines if x.startswith(('PIPE ','WORK ','LATENCY ','ROUTE ','BUILD ')))
 blob=base64.b85encode(lzma.compress(detail.encode(),preset=9)).decode()
 return ['META S14 f1c77ce6 vs wrong-math PDIA;16 routed seeds M8/16/32. PIPE_XZ85 contains all96 pipe probes,66 work rows,6 latency rows and route counts,losslessly packed for the receipt cap.',
  'META Decode PIPE_XZ85 with dev/nt1_timeline/read_pipe.py;local timings are not H200 evidence.']+[x for x in lines if x.startswith(('CELL ','MATCHED_PHASE ','GATE '))]+['PIPE_XZ85 '+blob]

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;lines=['META P3a pipe identification,exact S14 f1c77ce6 math69;58.7us belongs toONCE,not S14. Diagnostic PDIA retains routedTMA/activationB and weight LDS,removes decoder/MMA with B-cache XOR checksums and a finite checksum tail;scalar epilogue remains. Wrong math only,no promotion.',
 'META microkernels: same runtime expert byte ranges+block multiplicity, TMA pure stream stages1/2/4,shared-only reads,full math fromHBM or4.7MiB hotL2,smem-only math;occupancy8/16/32warps via queried cap,4/8/16warp CTA geometries;ONCE-style NB2/NB4 activation gathers;ILP2 MMA chains. Single-op limits+dependence at ILP1/4,warp-count targets8/16/32/64;actual grid/residency/regs/spills and activeSM reported. No counters inferred from static issue totals.',
 'META latency36KiB TMA completion measured by immediate mbar wait afterissue,cold unique andhot<=2.4MiB,1/132/264CTAs;full transfer latency includes TMA/barrier overhead. Little law prints4TB/s/SM*latency against staged capacity vs one-consumed-slot lookahead;repeated-block traffic factor shown separately;not claimed HBM counters.16seed S14/PDIA router+MoE ABBA readflush,A:A;4seed micro matrix andlatencies.']
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in [pool.submit(fn) for fn in (smallm_module,pdia_module,pclk_module,pipe_module,activity_module)]:f.result()
 mod=pipe_module();sms=torch.cuda.get_device_properties(0).multi_processor_count;out=torch.zeros((sms*8*512,),device='cuda',dtype=torch.int32);stamps=torch.zeros((sms*8,8),device='cuda',dtype=torch.int64)
 lines.append(f'BUILD seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
 lines+=pipe_op_table(mod,out,stamps,sms,deadline)
 pairs={m:[] for m in (8,16,32)};aa={m:[] for m in pairs};micro={};lat={};phase={};worst=(0.,0.);done=0;peak=0.
 for seed in range(16):
  if deadline-time.monotonic()<max(10,peak*1.2+2):lines.append(f'BUDGET before_seed={seed}');break
  start=time.monotonic();w=rg_generate_weights(5353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];base=nt1_banks(w)
  st={k:base[k] for k in ('pnw','gs13','gs2','sms')};st['banks']={'lop3':base['bank']};smallm_bind(st,{8:69});pdia_bind(st,{8:69,16:325});pclk_bind(st,{8:69,16:325});bank=base['bank']
  for m in pairs:
   pr,rs=routed_fixture(w,m,9000+m+97*seed);truth={};a=TimelineGraph(lambda:pipe_routed(pr,rs,st,'S14'));b=TimelineGraph(lambda:pipe_routed(pr,rs,st,'PDIA'));worst=max(worst,reuse_gate(a.replay(),w,truth,pr,rs))
   x,_,_,lids=a.replay()[1:];counts=torch.bincount(lids[lids>=0].long(),minlength=32).cpu().tolist();U=sum(n>0 for n in counts)*28311552;rho=sum((n+7)//8 for n in counts)/max(sum(n>0 for n in counts),1)
   ta,tb,r=timeline_timing(a,b);na=pipe_profile(a,'moe_var');nb=pipe_profile(b,'moe_var');pairs[m].append((ta,tb,r,na,nb,U/na/1e6));ctrl=TimelineGraph(lambda:pipe_routed(pr,rs,st,'S14'));v0,v1,_=timeline_timing(a,ctrl);aa[m].append(v1-v0)
   if seed<8:
    grid=min(264,sms*pclk_module().epi_info(325,2,m)[0]);trace=torch.zeros((grid,64),device='cuda',dtype=torch.int64)
    off=TimelineGraph(lambda:pipe_routed(pr,rs,st,'PCLK'));tr=TimelineGraph(lambda:pipe_routed(pr,rs,st,'PCLK',True,trace));reuse_gate(off.replay(),w,truth,pr,rs);reuse_gate(tr.replay(),w,truth,pr,rs);o,tax,ratio=timeline_timing(off,tr);tr.replay();torch.cuda.synchronize();v=trace.cpu().tolist();crit=max(v,key=lambda r:r[8]);phase.setdefault(m,[]).append((ratio,tax-o,crit[61]/1000,crit[62]/1000,crit[63]/1000,crit[52]/1000,crit[53]/1000,crit[16]/1000,(crit[7]-crit[5])/1000))
   if seed<4:
    ids=pipe_ids(counts,min(264,sms*2))
    if not ids.numel():lines.append(f'ZERO_EXPERT_WORK M{m} seed{seed}');continue
    ptr=[bank['w13p'].data_ptr(),bank['w2p'].data_ptr(),ids.data_ptr(),x.data_ptr(),out.data_ptr(),stamps.data_ptr(),torch.cuda.current_stream().cuda_stream]
    names=['COPY_S1','COPY_S2','COPY_S4','WEIGHT_LDS','MATH_HBM','MATH_L2','MATH_SMEM_W8','MATH_SMEM_W4','MATH_SMEM_W16','GATHER_NB2','GATHER_NB4','MMA_ILP2','B_ONLY_NB4','B_ONLY_NB2']
    for kind,name in enumerate(names):
     if (kind in (9,13) and m<16) or (kind in (10,12) and m<32):continue
     for cap in ((1,2,4) if kind in (0,1,6,7,8) else (1,) if kind in (9,10,12,13) else (2,)):
      geom=pipe_call(mod.pipe_work_run,ptr,kind,ids.numel(),m,cap,1)
      it=max(8,round(ids.numel()/max(1,geom[0])*(2 if kind==7 else 1)))
      g=PipeGraph(lambda:pipe_call(mod.pipe_work_run,ptr,kind,ids.numel(),m,cap,it));node=pipe_profile(g);g.replay();torch.cuda.synchronize();G=g.geometry[0];rows=stamps[:G].cpu().tolist();cycles=statistics.median(v[3]-v[2] for v in rows);tiles=sum(v[4] for v in rows);bytes_=tiles*g.geometry[5]
      assert G>0 and tiles>0 and all(v[3]>v[2] for v in rows)
      if kind<6:assert tiles==ids.numel(),(kind,tiles,ids.numel())
      # HOT smem math reports equivalent packed-byte rate;not off-chip bytes. Gather NB modes do more MMA.
      clk=statistics.median((v[3]-v[2])/(v[1]-v[0]) for v in rows if v[1]>v[0]);wait=statistics.mean(v[5]/max(v[4],1) for v in rows)
      micro.setdefault((m,name,cap),[]).append((node,bytes_/node/1e6,clk,cycles,wait,g.geometry))
    if seed==0:lines.append(f'ROUTE M{m} unique_bytes={U} repeated_block_factor={rho:.5f} counts={counts}')
  if seed<4:
   for hot in (0,1):
    for grid in (1,sms,sms*2):
     latbuf=torch.zeros((grid*32,4),device='cuda',dtype=torch.int64);ptr=[bank['w13p'].data_ptr(),bank['w2p'].data_ptr(),latbuf.data_ptr(),torch.cuda.current_stream().cuda_stream]
     g=PipeGraph(lambda:pipe_call(mod.pipe_lat_run,ptr,grid,36864,32,hot));pipe_time(g);lab_cold_l2();g.replay();torch.cuda.synchronize();v=latbuf.cpu().tolist();us=[(r[1]-r[0])/1000 for r in v];ghz=statistics.median((r[3]-r[2])/(r[1]-r[0]) for r in v if r[1]>r[0]);lat.setdefault((hot,grid),[]).append((statistics.median(us),ghz))
  done+=1;peak=max(peak,time.monotonic()-start)
 for m,vs in pairs.items():
  lines.append(f'CELL M{m} S14={statistics.mean(v[0] for v in vs):.3f} PDIA={statistics.mean(v[1] for v in vs):.3f} ratio={statistics.mean(v[2] for v in vs):.5f} S14_node_us={statistics.mean(v[3] for v in vs):.3f} PDIA_node_us={statistics.mean(v[4] for v in vs):.3f} unique_TBs={statistics.mean(v[5] for v in vs):.3f} n={len(vs)} AAabs={statistics.mean(map(abs,aa[m])):.3f}')
 for m,vs in phase.items():lines.append(f'MATCHED_PHASE M{m} n={len(vs)} observer,tax_us,Bfill_us,W13math_us,W2math_us,W13storebar_us,W2storebar_us,ready_us,grid_us='+','.join(f'{statistics.mean(v[j] for v in vs):.4f}' for j in range(9)))
 for (m,name,cap),vs in sorted(micro.items()):lines.append(f'WORK M{m} {name} cap{cap} n={len(vs)} node_us={statistics.mean(v[0] for v in vs):.3f} packed_TBs={statistics.mean(v[1] for v in vs):.3f} clock_GHz={statistics.mean(v[2] for v in vs):.4f} CTAcycles={statistics.mean(v[3] for v in vs):.1f} exposed_wait_cycles_per_tile={statistics.mean(v[4] for v in vs):.2f} G/res/regs/local/shared/bytes/warps={vs[0][5]}')
 for (hot,grid),vs in sorted(lat.items()):
  us=statistics.mean(v[0] for v in vs);need=4e6/sms*us;lines.append(f'LATENCY hot{hot} grid{grid} n={len(vs)} TMA36KiB_us={us:.4f} clock_GHz={statistics.mean(v[1] for v in vs):.4f} required_bytes_per_SM_at4TBs={need:.1f} stage_capacity2CTA2slots={2*2*36864} steady_lookahead_one_live_slot_perCTA={2*36864} steady_coverage={2*36864/need:.3f}')
 lines.append(f'GATE S14 worst={worst};wrong-math PDIA deliberately ungated;END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f}')
 return compact_pipe(lines)
