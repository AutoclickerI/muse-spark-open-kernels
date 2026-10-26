"""One MoE-host verdict for two-CTA compact scratch and3full raw slots."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent.parent;R=D.parents[5];sys.path.insert(0,str(D.parent));import build
text=(R/'prof/lab/moe_nt1_reuse.py').read_text();_,cell,_=build.review.original();text+='\n'+build.rename(cell,'rv0')
cells={**{k:v for k,v in json.loads((D.parent/'cells.json').read_text()).items() if k.startswith('FUSED')},**json.loads((D/'cells.json').read_text())};manifest=json.loads((D/'manifest.json').read_text())
for spec in cells.values():
 s=(R/spec['cell']).read_text();assert build.sha(s)==spec['cell_sha256'];text+='\n'+s
text+='\n_struct_original_experts=rv0_experts\n_occ_original_experts=rv0_experts\n';text+='\n'+(D.parent/'base_selector.py.inc').read_text()
for arm in manifest:text+='\n'+(D/(arm.lower()+'_selector.py.inc')).read_text()
old=(P/'review/lab.py.inc').read_text();a=old.index('def review_gate');b=old.index('\ndef review_nodes',a)
body='''STRUCT_ARMS={'BASE':'rvc_struct_base','SINGLE2':'rvc_occ_single2','SINGLE3':'rvc_occ_single3','HALF2':'rvc_occ_half2','HALF3':'rvc_occ_half3','DOUBLE2':'rvc_occ_double2','DOUBLE3':'rvc_occ_double3'}
def review_call(pr,rs,state,arm):
 x,ids,tw,lids=route_call(pr,rs);d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 tag='rv0' if arm=='S15' else STRUCT_ARMS[arm]
 return globals()[tag+'_experts'](d,state,69),x,ids,tw,lids
'''+old[a:b]+'''
def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 begin=time.monotonic();deadline-=5;rows=[];aa_rows=[];gates={};failures={};done=0;peak=0.;arm_names=['BASE','S15','SINGLE2','SINGLE3','HALF2','HALF3','DOUBLE2','DOUBLE3'];resources=[]
 fns=[rv0_module,rvc_struct_fused0_module,rvc_struct_fused1_module,rvc_occ_single0_module,rvc_occ_single1_module,rvc_occ_half0_module,rvc_occ_half1_module,rvc_occ_double0_module,rvc_occ_double1_module]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(fn) for fn in fns]:f.result()
 build_seconds=time.monotonic()-begin
 for seed in range(16):
  if deadline-time.monotonic()<max(12,peak*1.25+2):break
  start=time.monotonic();w=rg_generate_weights(675353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w)
  state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']};rv0_bind(state,{8:69})
  for name in ('fused0','fused1'):globals()['rvc_struct_'+name+'_bind'](state,{8:69})
  for name in ('single0','single1','half0','half1','double0','double1'):globals()['rvc_occ_'+name+'_bind'](state,{8:69})
  for mm in (8,16,32):
   if seed==0:
    ctrl=globals()['rvc_struct_'+('fused0' if mm==8 else 'fused1')+'_module']()
    infos=[[name,[globals()['rvc_occ_'+name+('0' if mm==8 else '1')+'_module']().epi_info(69,depth,mm) for depth in (2,3)]] for name in ('single','half','double')]
    resources.append([mm,ctrl.epi_info(69,2,mm),infos])
   pr,rs=routed_fixture(w,mm,679000+mm+97*seed);truth={};graphs={}
   for arm in arm_names:
    if arm in failures:continue
    try:
     g=TimelineGraph(lambda arm=arm:review_call(pr,rs,state,arm));err=review_gate(g.replay(),w,truth,pr,rs);graphs[arm]=g;gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0.,0.]),err)]
    except Exception as e:failures[arm]=f'M{mm} seed{seed} '+repr(e)
   if 'BASE' not in graphs:continue
   ctrl=TimelineGraph(lambda:review_call(pr,rs,state,'BASE'));aa,aa2,aar=timeline_timing(graphs['BASE'],ctrl);aa_rows.append([seed,mm,round(aa*1000),round(aa2*1000),round(aar*1e6)])
   order=arm_names[1:];order.reverse() if seed%2 else None
   for arm in order:
    if arm not in graphs:continue
    old,new,ratio=timeline_timing(graphs['BASE'],graphs[arm]);err=review_gate(graphs[arm].replay(),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates[arm],err)]
    rows.append([seed,mm,arm_names.index(arm),0,round(old*1000),round(new*1000),round(ratio*1e6)])
   if 'S15' in graphs:
    for arm in arm_names[2:]:
     if arm in graphs:
      old,new,ratio=timeline_timing(graphs['S15'],graphs[arm]);rows.append([seed,mm,arm_names.index(arm),1,round(old*1000),round(new*1000),round(ratio*1e6)])
  done+=1;peak=max(peak,time.monotonic()-start)
 record=dict(base='c49 PAIR M8;PAIR+BLOCK M16/32',seeds=done,requested_seeds=16,arm_names=arm_names,rows=rows,AA_rows=aa_rows,gates=gates,failures=failures,build_seconds=build_seconds,elapsed_seconds=time.monotonic()-begin)
 packed=review_pack(record)
 lines=['META P3a3fullrawslots keeping2CTA: SINGLE single4KiB reduction+routealias;HALF single2KiB paired reduction+eflatalias;DOUBLE2x2KiB paired reduction+routealias. Each depth2/3 versus c49BASE/S15;16freshseeds M8/16/32,624pairs48AA,TASK/readflushABBA.',
 'META SINGLE/DOUBLE alias dead route counts,pack reverse indices8bit andoffset16bit,rebuild inverse route after math. HALF keeps inverse route. Added barriers enforce lifetimes. Local counts predict2CTAs,121-125regs0spills at3slots;actualruntime occupancy below. Every pointwin to P4,R confirms.',
 f'GATE seeds={done}/16 failure_count={len(failures)} max_total={max(v[0] for v in gates.values()):.7f} max_row={max(v[1] for v in gates.values()):.7f}',
 'RESOURCE [M,BASE,[kind,[depth2,depth3](resident,regs,local,dynamic,static)]] '+repr(resources),
 'P3A_REVIEW_XZ85 '+packed,
 f'END seeds={done} seconds={time.monotonic()-begin:.1f};no whole-layer promotion']
 assert sum(len(l)+1 for l in lines)<9000
 return lines
'''
codec=(P/'review/receipt.py').read_text().replace("weights_seed='175353+7919*seed',route_seed='179000+M+97*seed'","weights_seed='675353+7919*seed',route_seed='679000+M+97*seed'")
text+='\n'+codec+'\n'+body;ast.parse(text);expanded=build.sha(text);blob=base64.b85encode(lzma.compress(text.encode(),preset=9)).decode();nbytes=len(text)
text='import base64,lzma,linecache,hashlib\n_p3a_body=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n'
text+='assert hashlib.sha256(_p3a_body.encode()).hexdigest()=='+repr(expanded)+'\n_p3a_file="<p3a-occupancy-"+'+repr(expanded[:12])+'+">"\n'
text+='linecache.cache[_p3a_file]=(len(_p3a_body),None,_p3a_body.splitlines(True),_p3a_file)\nexec(compile(_p3a_body,_p3a_file,"exec"),globals())\n_p3a_lab=lab\ndef lab(deadline,weights):\n    return _p3a_lab(deadline,weights)\n'
p=R/'prof/lab/moe_p3a_occupancy.py';p.write_text(text);(D/'payload.json').write_text(json.dumps(dict(payload=str(p.relative_to(R)),sha256=build.sha(text),expanded_sha256=expanded,expanded_bytes=nbytes,packed_bytes=len(text),seeds=16,M=[8,16,32],max_pairs=624,AA_pairs=48,weights_seed='675353+7919*seed',route_seed='679000+M+97*seed'),indent=2)+'\n');print(p,len(text),expanded)
