BAL_NAMES=('S15','CH4','B8','B4','B2','B0','ONE')
BAL_NS={}
def balance_build():
 import linecache
 if BAL_NS:return
 nf_module();NF_OPTS['pmz']=nf_module().pmz_init();BAL_NS['S15']=globals()
 for name,cell in [('CH4',CHUNK_CELL),('BAL',BALANCE_CELL)]:
  ns=dict(globals());filename='<balance-'+name+'>';linecache.cache[filename]=(len(cell),None,cell.splitlines(True),filename)
  exec(compile(cell,filename,'exec'),ns)
  if name=='CH4':ns['p3b_module']();ns['p3b_module']().ch_info(4,0);ns['p3b_module']().ch_info(4,1)
  else:ns['balance_module']()
  BAL_NS[name]=ns

def balance_prepare(w):
 w=dict(w,post_norm_w=w['post_expert_norm'])
 base=nf_perm(nf_wide(m128_hh_issue_prepare(nf_ilv(w))));native=m128_hh_issue_nt_prepare(w)
 streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event())
 return dict(base=base,native=native,streams=streams)

def balance_plan_for(data,state,arm):
 args=(data,state['base'],state['native'],state['streams'])
 if arm=='S15':return nf_plan(*args)
 if arm=='CH4':return BAL_NS['CH4']['p3b_plan'](*args)
 return BAL_NS['BAL']['balance_plan'](*args,arm)

def balance_call(data,state,arm):
 p=balance_plan_for(data,state,arm)
 for f in p['steps']:out=f()
 return out

def oracle_module():
 if not hasattr(oracle_module,'mod'):
  arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
  oracle_module.mod=load_inline(name='p3b_stream_'+hashlib.sha256(ORACLE_CUDA.encode()).hexdigest()[:12],cpp_sources='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nvoid stream_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=ORACLE_CUDA,functions=['stream_run'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
 return oracle_module.mod

def oracle_graph(data,state,mode,tile,split=False):
 # Real math outside capture initializes W2's activation buffer. Timed oracle is
 # checksum streaming only; route/finish are excluded consistently in U/W/X/B.
 p=balance_plan_for(data,state,'S15');[f() for f in p['steps']];torch.cuda.synchronize()
 h13,h2,nv=p['owners'][:3];m=data['hidden_states'].shape[0]
 sms=torch.cuda.get_device_properties(0).multi_processor_count;mod=nf_module();stages=2
 while stages>1 and mod.pmz_smem(stages,m)>NF_OPTS['pmz']:stages-=1
 g2,g13=[sms*mod.pmz_resident(stages,m,f) for f in (0,1)]
 output=torch.empty((3*sms*8*256,),device='cuda',dtype=torch.int32)
 tensors=[state['native']['w13p'],state['native']['w2p'],h13[3],h2[3],h13[4],h13[5],h13[8],output]
 counts=h13[4].cpu().tolist();metadata={};work={}
 for mask,phase in ((7,-1),(1,1),(1,0),(4,1),(4,0),(2,1),(2,0)):
  wx=[];ax=[];prefix=[0]
  for e,n in enumerate(counts):
   ww=[];aa=[];branch=1 if n>16 else 4 if n>8 else 2
   active=n>0 and bool(mask&branch)
   for f in (0,1):
    nt=512 if f else 256;rep=(n+tile-1)//tile if mode and n>16 else 1;ar=0
    if active and mode==2:
     if n>8:ar=(128 if f else 64)//(2 if n>16 and tile==64 else 1)
     else:
      total=sum(0<c<=8 for c in counts)*nt;begin=sum(0<c<=8 for c in counts[:e])*nt;grid=g13 if f else g2
      ar=sum(total*c//grid<total*(c+1)//grid and total*(c+1)//grid>begin and total*c//grid<begin+nt for c in range(grid))
    ww.append(nt*rep if active and (phase<0 or phase==f) else 0)
    aa.append(n*ar if active and (phase<0 or phase==f) else 0)
   wx+=ww;ax+=aa;prefix.append(prefix[-1]+sum(ww)+sum(aa))
  metadata[mask,phase]=torch.tensor(prefix+wx+ax,device='cuda',dtype=torch.int32)
  work[mask,phase]=dict(weight_MB=sum(wx)*.036864,valid_X_MB=sum(ax)*.008192,checksum_store_MB=sms*8*256*4/1e6)
 tensors.append(metadata[7,-1])
 v=[t.data_ptr() for t in tensors];om=oracle_module()
 side,fork,join,light,ljoin=state['streams']
 def run():
  _=(p,tensors,metadata);parent=torch.cuda.current_stream()
  if not split:om.stream_run(v,m,mode,tile,7,-1,g13,g2,sms*8,parent.cuda_stream)
  else:
   fork.record(parent);side.wait_event(fork);light.wait_event(fork)
   for stream,mask,event in ((side,1,join),(light,4,ljoin)):
    with torch.cuda.stream(stream):
     vv=list(v);vv[7]+=4*sms*8*256*(1 if mask==1 else 2)
     for f in (1,0):
      vv[8]=metadata[mask,f].data_ptr();om.stream_run(vv,m,mode,tile,mask,f,g13,g2,sms*8,stream.cuda_stream)
     event.record()
   for f in (1,0):
    vv=list(v);vv[8]=metadata[2,f].data_ptr();om.stream_run(vv,m,mode,tile,2,f,g13,g2,sms*8,parent.cuda_stream)
   parent.wait_event(join);parent.wait_event(ljoin)
  return output
 # No data refresh: one specified fixture is the oracle's immutable byte-address
 # workload; this diagnostic returns no math output to a host candidate.
 graph=MWGraph(lambda _:run(),data);graph.owners=(p,tensors,metadata,run);graph.work=work[7,-1]
 return graph

def clock_plan(data,state,arm):
 # Events delimit actual concurrent branches, with all start offsets measured
 # from the route completion. A separately timed unmarked graph measures observer cost.
 import linecache
 ns=dict(BAL_NS['CH4'] if arm=='CH4' else globals())
 fn=ns['p3b_plan'] if arm=='CH4' else nf_plan
 src=CLOCK_CH4 if arm=='CH4' else CLOCK_S15;marks={k:torch.cuda.Event(enable_timing=True,external=True) for k in ('origin','h13','h2','n13','n2','l13','l2','joined','end')}
 ns['_marks']=marks
 src=src.replace('fork.record(parent);','_marks["origin"].record(parent);fork.record(parent);')
 if arm=='CH4':
  src=src.replace('mod.ch_run(p2,0,4,','_marks["h13"].record();mod.ch_run(p2,0,4,')
 else:src=src.replace('mod.br_run(p2,21,','_marks["h13"].record();mod.br_run(p2,21,')
 src=src.replace(';join.record()', ';_marks["h2"].record();join.record()')
 src=src.replace('mod.br_run(pl2,71,','_marks["l13"].record();mod.br_run(pl2,71,')
 src=src.replace(';ljoin.record()', ';_marks["l2"].record();ljoin.record()')
 src=src.replace('mod.pmz_run(pn,0,8,','_marks["n13"].record();mod.pmz_run(pn,0,8,')
 src=src.replace('        parent.wait_event(join);','        _marks["n2"].record();parent.wait_event(join);')
 src=src.replace('parent.wait_event(ljoin)','parent.wait_event(ljoin);_marks["joined"].record()')
 src=src.replace(';return out',';_marks["end"].record();return out')
 filename='<clock-'+arm+'>';linecache.cache[filename]=(len(src),None,src.splitlines(True),filename);exec(compile(src,filename,'exec'),ns)
 def call(d):
  p=ns[fn.__name__](d,state['base'],state['native'],state['streams'])
  for f in p['steps']:out=f()
  return out
 g=MWGraph(call,data);g.marks=marks
 return g
