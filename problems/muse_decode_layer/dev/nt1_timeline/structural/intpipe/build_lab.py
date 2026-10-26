"""One structured MoE check:exact S19 cells,two independent address arms,all seeds."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];cells=json.loads((D/'cells.json').read_text());manifest=json.loads((D/'manifest.json').read_text());text=(R/'prof/lab/moe_nt1_reuse.py').read_text()
for v in cells.values():text+='\n'+(R/v['cell']).read_text()
old=(D.parent.parent/'review/lab.py.inc').read_text();text+='\n'+old[old.index('def review_gate'):old.index('\ndef review_nodes')]
text+='''
def ip_call(pr,rs,state,arm):
 x,ids,tw,lids=route_call(pr,rs);mm=x.shape[0];i=0 if mm==1 else 1 if mm==8 else 2
 tag='ipc_'+('base' if arm=='AFFINE' and mm==1 else arm.lower())+str(i)
 out=globals()[tag+'_experts'](dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0}),state,95 if mm==1 else 69)
 return out,x,ids,tw,lids

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();names=['BASE','AFFINE','PRIVATE'];paired=[];aa=[];gates={};failures={};resources=[];done=0;peak=0.
 mods=[globals()['ipc_'+a+str(i)+'_module'] for a in ('base','affine','private') for i in range(3) if not(a=='affine' and i==0)]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(fn) for fn in mods]:f.result()
 build_s=time.monotonic()-begin
 for seed in range(16):
  if deadline-time.monotonic()<max(12,1.25*peak+2):break
  start=time.monotonic();w=rg_generate_weights(975353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w);state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']}
  for a in ('base','affine','private'):
   for i in range(3):
    if a=='affine' and i==0:continue
    globals()['ipc_'+a+str(i)+'_bind'](state,{1:95} if i==0 else {8:69})
  for m in (1,8,16,32):
   i=0 if m==1 else 1 if m==8 else 2
   if seed==0:
    for a in names:
     tag='ipc_'+('base' if a=='AFFINE' and m==1 else a.lower())+str(i);resources.append([m,a,globals()[tag+'_module']().epi_info(95 if m==1 else 69,2,m)])
   pr,rs=routed_fixture(w,m,979000+m+97*seed);truth={};graphs={}
   for arm in names:
    try:
     g=TimelineGraph(lambda arm=arm:ip_call(pr,rs,state,arm));err=review_gate(g.replay(),w,truth,pr,rs);graphs[arm]=g;gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0.,0.]),err)]
    except Exception as e:failures.setdefault(arm,[]).append([seed,m,repr(e)])
   if 'BASE' not in graphs:continue
   ctrl=TimelineGraph(lambda:ip_call(pr,rs,state,'BASE'));t1,t2,ratio=timeline_timing(graphs['BASE'],ctrl);aa.append([seed,m,round(t1*1000),round(t2*1000),ratio])
   for arm in (names[1:] if seed%2==0 else names[:0:-1]):
    if arm not in graphs:continue
    old,new,ratio=timeline_timing(graphs['BASE'],graphs[arm]);err=review_gate(graphs[arm].replay(),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates[arm],err)];paired.append([seed,m,arm,old,new,ratio])
  done+=1;peak=max(peak,time.monotonic()-start)
 return {'packet':dict(kind='P3A_INTPIPE',identities=IP_IDENTITIES,seeds=dict(requested=16,completed=done,weights='975353+7919*seed',routes='979000+M+97*seed'),paired=paired,AA=aa,resources=resources,checks=dict(gates=gates,failures=failures),complete=done==16 and len(paired)==128 and len(aa)==64 and not failures,build_seconds=build_s,elapsed_seconds=time.monotonic()-begin,control_aliases=[['AFFINE',1]],interpretation='Stage nomination only;P4 composes against S19 and confirms held-out six-load R')}
'''
identities=dict(base_S19='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a',cells={k:{f:v[f] for f in ('cell_sha256','cuda_sha256')} for k,v in cells.items()},arms={a:{k:v[k] for k in ('candidate_sha256','patch_sha256')} for a,v in manifest.items()})
text+='\nIP_IDENTITIES='+repr(identities)+'\n';ast.parse(text);expanded=hashlib.sha256(text.encode()).hexdigest();blob=base64.b85encode(lzma.compress(text.encode(),preset=9)).decode()
packed='''LAB_RESOURCES={'compile_s':70,'run_s':70,'output_bytes':8744,'context_poison':'none'}
import base64,lzma,linecache,hashlib
_source=lzma.decompress(base64.b85decode('''+repr(blob)+''')).decode()
assert hashlib.sha256(_source.encode()).hexdigest()=='''+repr(expanded)+'''
_file='<p3a-intpipe-'''+expanded[:12]+'''>'
linecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)
exec(compile(_source,_file,'exec'),globals())
_ip_lab=lab
def lab(deadline,weights):return _ip_lab(deadline,weights)
'''
p=R/'prof/lab/moe_p3a_intpipe.py';p.write_text(packed);(D/'payload.json').write_text(json.dumps(dict(payload=str(p.relative_to(R)),sha256=hashlib.sha256(packed.encode()).hexdigest(),expanded_sha256=expanded,identities=identities),indent=2)+'\n');print('payload',len(packed),'expanded',len(text),expanded)
