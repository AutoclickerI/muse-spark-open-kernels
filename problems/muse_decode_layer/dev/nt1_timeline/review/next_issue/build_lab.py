"""Pack three independently screened instruction cuts against the measured per-M parent."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
D=Path(__file__).resolve().parent;V=D.parent;R=D.parents[5];sys.path.insert(0,str(D));import build
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
text=(R/'prof/lab/moe_nt1_reuse.py').read_text();_,cell,_=build.review.original();text+='\n'+build.rename(cell,'rv0')
m=json.loads((D/'manifest.json').read_text())
for arm,spec in m.items():
 for v in spec['variants']:
  cell=(R/v['cell']).read_text();assert sha(cell)==v['cell_sha256'];text+='\n'+cell
 text+='\n'+(D/(arm.lower()+'_selector.py.inc')).read_text()
body='''NEXT_ARMS={'BASE':'rvc_nbase','FAST':'rvc_nfast','INDEX':'rvc_nindex','FLUSH1':'rvc_nflush1'}
def review_call(pr,rs,state,arm):
 x,ids,tw,lids=route_call(pr,rs);m=x.shape[0];d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 tag='rv0' if arm=='S15' else NEXT_ARMS[arm]
 return globals()[tag+'_experts'](d,state,95 if m==1 else 69),x,ids,tw,lids
'''
old=(V/'lab.py.inc').read_text();a=old.index('def review_gate');b=old.index('\ndef review_nodes',a);body+='\n'+old[a:b]
body+='''
def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;rows=[];aa_rows=[];gates={};failures={};done=0;peak=0.;arm_names=['BASE','S15','FAST','INDEX','FLUSH1']
 fns=[rv0_module]+[globals()[tag+'_module'] for tag in NEXT_ARMS.values()]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(fn) for fn in fns]:f.result()
 build_seconds=time.monotonic()-begin
 for seed in range(16):
  if deadline-time.monotonic()<max(12,peak*1.25+2):break
  start=time.monotonic();w=rg_generate_weights(375353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w)
  state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']};rv0_bind(state,{1:95,8:69})
  for arm,tag in NEXT_ARMS.items():
   if arm in failures:continue
   try:globals()[tag+'_bind'](state,{1:95,8:69})
   except Exception as e:failures[arm]='bind '+repr(e)
  for mm in (1,8,16,32):
   pr,rs=routed_fixture(w,mm,379000+mm+97*seed);truth={};graphs={}
   for arm in arm_names:
    if arm in failures:continue
    try:
     g=TimelineGraph(lambda arm=arm:review_call(pr,rs,state,arm));err=review_gate(g.replay(),w,truth,pr,rs);graphs[arm]=g;gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0.,0.]),err)]
    except Exception as e:failures[arm]=f'M{mm} seed{seed} '+repr(e)
   if 'BASE' not in graphs:continue
   ctrl=TimelineGraph(lambda:review_call(pr,rs,state,'BASE'));aa,aa2,aar=timeline_timing(graphs['BASE'],ctrl);aa_rows.append([seed,mm,round(aa*1000),round(aa2*1000),round(aar*1e6)])
   order=['S15','FAST','INDEX','FLUSH1'];order.reverse() if seed%2 else None
   for arm in order:
    if arm not in graphs:continue
    old,new,ratio=timeline_timing(graphs['BASE'],graphs[arm]);err=review_gate(graphs[arm].replay(),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates[arm],err)]
    rows.append([seed,mm,arm_names.index(arm),0,round(old*1000),round(new*1000),round(ratio*1e6)])
   if 'S15' in graphs:
    for arm in ('FAST','INDEX','FLUSH1'):
     if arm in graphs:
      old,new,ratio=timeline_timing(graphs['S15'],graphs[arm]);rows.append([seed,mm,arm_names.index(arm),1,round(old*1000),round(new*1000),round(ratio*1e6)])
  done+=1;peak=max(peak,time.monotonic()-start)
 record=dict(base='c49 measured SHARD2 M1,PAIR M8,PAIR+BLOCK M16/M32',seeds=done,requested_seeds=16,arm_names=arm_names,rows=rows,AA_rows=aa_rows,gates=gates,failures=failures,build_seconds=build_seconds,elapsed_seconds=time.monotonic()-begin)
 packed=review_pack(record)
 lines=['META P3a measured c49 parent versus independent FAST normalized FP32 cancellation,INDEX bounded vector addresses/hoisted B-fill branches,FLUSH1 remove duplicate zero norm flush. S15 also paired.',
 'META16 fresh seeds/M1,8,16,32; TASK only,before/after;64 independent BASE:A controls; swapped-order read-flushed pairs. Every point win nominates; P4 held-out six-load R confirms.',
 f'GATE seeds={done}/16 failure_count={len(failures)} max_total={max(v[0] for v in gates.values()):.7f} max_row={max(v[1] for v in gates.values()):.7f}',
 'P3A_REVIEW_XZ85 '+packed,
 f'END seeds={done} seconds={time.monotonic()-begin:.1f};no whole-layer promotion']
 assert sum(len(l)+1 for l in lines)<9000
 return lines
'''
codec=(V/'receipt.py').read_text().replace("weights_seed='175353+7919*seed',route_seed='179000+M+97*seed'","weights_seed='375353+7919*seed',route_seed='379000+M+97*seed'")
text+='\n'+codec+'\n'+body;ast.parse(text);digest=sha(text);expanded_bytes=len(text);blob=base64.b85encode(lzma.compress(text.encode(),preset=9)).decode()
text='import base64,lzma,linecache,hashlib\n_p3a_body=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n'
text+='assert hashlib.sha256(_p3a_body.encode()).hexdigest()=='+repr(digest)+'\n_p3a_file="<p3a-next-"+'+repr(digest[:12])+'+">"\n'
text+='linecache.cache[_p3a_file]=(len(_p3a_body),None,_p3a_body.splitlines(True),_p3a_file)\nexec(compile(_p3a_body,_p3a_file,"exec"),globals())\n_p3a_lab=lab\ndef lab(deadline,weights):\n    return _p3a_lab(deadline,weights)\n'
path=R/'prof/lab/moe_p3a_next_issue.py';path.write_text(text)
(D/'payload.json').write_text(json.dumps(dict(payload=str(path.relative_to(R)),sha256=sha(text),expanded_sha256=digest,expanded_bytes=expanded_bytes,packed_bytes=len(text),arms=m,requested_seeds=16,weights_seed='375353+7919*seed',route_seed='379000+M+97*seed',max_pairs=448,max_AA_pairs=64),indent=2)+'\n');print(path,len(text),digest)
