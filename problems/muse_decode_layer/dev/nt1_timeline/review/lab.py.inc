REVIEW_ARMS={'BLOCK':'rvb','CANCEL':'rvc','PAIR':'rvp','SHARD2':'rvs2','SHARD4':'rvs4','SHARD8':'rvs8','PACK5':'rvq'}

def review_call(pr,rs,state,arm):
 x,ids,tw,lids=route_call(pr,rs);m=x.shape[0];d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 flags=95 if m==1 else 69
 tag='rv0' if arm=='S15' or (arm=='BLOCK' and m not in (16,32)) else REVIEW_ARMS[arm]
 out=globals()[tag+'_experts'](d,state,flags)
 return out,x,ids,tw,lids

def review_gate(output,w,truth,pr,rs):
 got,x,ids,tw,lids=output
 _,_,biased=route_reference(pr,rs);route_gate(ids,biased,int(rs['start']))
 key=tuple(ids.flatten().cpu().tolist())
 if key not in truth:truth[key]=nt1_fp32(dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0}),w)
 ok,total,row=task_check(got,truth[key])
 if not ok:raise ValueError(f'TASK MoE {total:.7g}/{row:.7g}')
 return total,row

def review_nodes(graph):
 mod=activity_module();mod.profile_begin()
 try:
  for _ in range(3):lab_cold_l2();graph.replay();torch.cuda.synchronize()
 finally:records=mod.profile_end()
 vals=[(v[1]-v[0])/1000 for k,v in records if 'moe_var' in k and v[4]>0][-3:]
 if len(vals)!=3:raise RuntimeError('missing consumer nodes')
 return statistics.median(vals)

def lab(deadline,weights):
 from concurrent.futures import ThreadPoolExecutor
 import base64,json,lzma
 begin=time.monotonic();deadline-=5
 fns=[rv0_module,activity_module]+[globals()[tag+'_module'] for tag in REVIEW_ARMS.values()]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [pool.submit(fn) for fn in fns]:f.result()
 build=time.monotonic()-begin;done=0;peak=0.;rows=[];aa_rows=[];gates={};failures={};coding=[];arm_names=['S15',*REVIEW_ARMS]
 for seed in range(16):
  if deadline-time.monotonic()<max(12,peak*1.25+2):break
  start=time.monotonic();w=rg_generate_weights(175353+7919*seed,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w)
  state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']};rv0_bind(state,{1:95,8:69})
  for arm,tag in REVIEW_ARMS.items():
   if arm in failures:continue
   try:globals()[tag+'_bind'](state,{1:95,8:69})
   except Exception as e:failures[arm]='bind '+repr(e)
  if 'packet5' in state['banks']:
   coding.append({k:dict(raw_bytes=bank['bank'][k].numel(),coded_bytes=v.numel()) for k,v in state['banks']['packet5'].items()})
  for m in (1,8,16,32):
   pr,rs=routed_fixture(w,m,179000+m+97*seed);truth={};graphs={}
   for arm in ('S15',*REVIEW_ARMS):
    if arm in failures:continue
    try:
     g=TimelineGraph(lambda arm=arm:review_call(pr,rs,state,arm));err=review_gate(g.replay(),w,truth,pr,rs);graphs[arm]=g;gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0.,0.]),err)]
    except Exception as e:failures[arm]=f'M{m} seed{seed} '+repr(e)
   if 'S15' not in graphs:continue
   ctrl=TimelineGraph(lambda:review_call(pr,rs,state,'S15'));aa,aa2,aar=timeline_timing(graphs['S15'],ctrl);aa_rows.append([seed,m,round(aa*1000),round(aa2*1000),round(aar*1e6)])
   order=list(REVIEW_ARMS);order.reverse() if seed%2 else None
   for arm in order:
    if arm not in graphs:continue
    old,new,ratio=timeline_timing(graphs['S15'],graphs[arm]);err=review_gate(graphs[arm].replay(),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates[arm],err)]
    rows.append([seed,m,arm_names.index(arm),0,round(old*1000),round(new*1000),round(ratio*1e6)])
   # Shards are one change from PAIR; keep direct comparisons as well as the requested S15 control.
   if 'PAIR' in graphs:
    for arm in ('SHARD2','SHARD4','SHARD8'):
     if arm in graphs:
      old,new,ratio=timeline_timing(graphs['PAIR'],graphs[arm]);rows.append([seed,m,arm_names.index(arm),arm_names.index('PAIR'),round(old*1000),round(new*1000),round(ratio*1e6)])
   if seed<4:
    for arm,g in graphs.items():
     try:rows.append([seed,m,arm_names.index(arm),-1,round(review_nodes(g)*1000)])
     except Exception as e:failures[arm]='node '+repr(e)
  done+=1;peak=max(peak,time.monotonic()-start)
 record=dict(base='S15 60254883; exact SMALLM_VAR_CUDA and95/69 map',seeds=done,requested_seeds=16,columns=['seed','M','arm_index','control_index (-1 means CUPTI node)','old_ns (or node_ns)','new_ns','ratio_ppm'],arm_names=arm_names,rows=rows,AA_columns=['seed','M','A_ns','A2_ns','ratio_ppm'],AA_rows=aa_rows,timing_quantization='event times rounded to1ns; per-seed median-of-ratios rounded to1ppm',gates=gates,failures=failures,scale_coding=coding,build_seconds=build,elapsed_seconds=time.monotonic()-begin,point_estimate_nomination=True,whole_layer_promotion='P4 held-out six-load R lower confidence bound >1',decoder_two_base='dropped before H200 for integer issue growth')
 packed=review_pack(record)
 # Entire seed-level matrix is retained losslessly; no per-shape A:A veto.
 return ['META P3a S15 independent BLOCK/CANCEL/PACKET5 and PAIR+SHARD2/4/8;16 fresh seeds,M1/8/16/32;same complete routed work/precision. A:A diagnostic; point estimates nominate,P4 confirms six-load R.',
 'META BLOCK uses 32x4 release counters; CANCEL removes W2 global scale with eps/a^2/sign/zero+extreme handling;SHARD2/4/8 only change PAIR norm slots;PACK5 uses exact256-scale packets with raw fallback and timed decode. No W13 scale cancellation.',
 f'GATE seeds={done}/16 failures={failures} max_total={max(v[0] for v in gates.values()):.7f} max_row={max(v[1] for v in gates.values()):.7f}; all per-arm gates in archive',
 'P3A_REVIEW_XZ85 '+packed,
 f'END seeds={done} seconds={time.monotonic()-begin:.1f} left={deadline-time.monotonic():.1f};nomination only,no six-load promotion']
