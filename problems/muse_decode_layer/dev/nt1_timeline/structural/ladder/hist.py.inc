def lab_hist(deadline,weights):
 import json,lzma,base64,hashlib
 start=time.monotonic();hist=[];times=[];telemetry=[];gates={};w=rg_generate_weights(1375353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=nt1_banks(w);state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']}
 for i in range(3):globals()['lad'+str(i)+'_bind'](state,{1:95,8:69})
 for M in (1,8,16,32):
  pr,rs=routed_fixture(w,M,1379000+M);x,ids,tw,lids=route_call(pr,rs);data=dict(hidden_states=x.contiguous(),topk_ids=lids.int().contiguous(),topk_weights=tw.float().contiguous());truth={};args={};graphs={}
  for arm in ('BASE','AA','HIST'):
   args[arm]=lad_args(data,state,arm);graphs[arm]=TimelineGraph(args[arm]['fn']);e=review_gate((graphs[arm].replay(),x,ids,tw,lids),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0,0]),e)]
  raw={a:[] for a in graphs};tel=[s21_telemetry()]
  for order in s21_orders(list(graphs),2,0):
   events=[]
   for arm in order:
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();graphs[arm].replay();b.record();events.append((arm,a,b))
   torch.cuda.synchronize()
   for arm,a,b in events:raw[arm].append(round(a.elapsed_time(b)*1e6))
   tel.append(s21_telemetry())
  h=lad_hist(args['HIST']['trace']);h['SM_columns']=[list(v) for v in zip(*h.pop('SM_rows'))];h['CTA_joint_columns']=[list(v) for v in zip(*h.pop('CTA_joint'))];hist.append([0,M,h]);times.append([M,raw]);telemetry.append([M,tel])
  for arm in graphs:
   e=review_gate((graphs[arm].replay(),x,ids,tw,lids),w,truth,pr,rs);gates[arm]=[max(a,b) for a,b in zip(gates.get(arm,[0,0]),e)]
 return dict(packet=dict(kind='P3A_LADDER_HIST',identities={'S19':LAD_IDENTITIES['base_S19'],'bundle':hashlib.sha256(json.dumps(LAD_IDENTITIES,sort_keys=True).encode()).hexdigest()},seeds=[0],checks=dict(gates=gates,failures=[]),complete=len(hist)==4,paired=dict(codec='lzma-json-b85',data=base64.b85encode(lzma.compress(json.dumps(dict(histograms=hist,round_ns=times,telemetry=telemetry),sort_keys=True,separators=(',',':')).encode(),preset=9)).decode()),elapsed_seconds=time.monotonic()-start,scope='Fixed extra seed0 diagnostic phase,not primary retiming;same code/weights/routes;primary HIST/BASE observer ratio is retained for all8seeds'))
