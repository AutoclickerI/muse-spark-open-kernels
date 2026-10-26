def overlap_trace(pr,rs,state,mm,which,stamps):
 x,ids,tw,lids=route_call(pr,rs);d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0});tag='rvc_ov_'+('clock' if which=='BASE' else 'ws8')+('0' if mm==8 else '1')
 return globals()[tag+'_experts'](d,state,325,stamps=stamps),x,ids,tw,lids

def overlap_frame(stamps,ratio):
 rows=stamps.cpu().tolist();active=[v for v in rows if v[128+7]>0];assert active,'trace has no math tiles'
 ghz=statistics.median((v[21]-v[20])/(v[9]-v[0]) for v in active if v[21]>v[20] and v[9]>v[0]);assert ghz>0
 mean=[]
 for w in range(12):
  mean.append([statistics.mean(v[128+w*10+k] for v in active)/(ghz if k<7 else 1) for k in range(10)])
 critical=max(active,key=lambda v:v[8]-v[0]);events=critical[64:112];origin=min(v for v in events if v>0);events=[(v-origin) if v else None for v in events]
 return dict(warps=mean,events=events,GHz=ghz,observer=ratio,critical_span_ns=critical[8]-critical[0],active_CTAs=len(active))

def overlap_diag_pack(records):
 import base64,struct
 words=[]
 for m in (8,16,32):
  for kind in ('BASE','WS8'):
   frames=records[(m,kind)];assert len(frames)==4
   for w in range(8 if kind=='BASE' else 9):
    for k in range(10):
     val=statistics.median(v['warps'][w][k] for v in frames);words.append(round(val/32) if k<7 else round(val))
   events=frames[0]['events'];words.extend(65535 if v is None else round(v/32) for v in events[:8]+events[16:24]+events[32:40])
   obs=[v['observer'] for v in frames];words.extend([round(statistics.mean(obs)*1000),round(statistics.stdev(obs)/2*10000),round(statistics.median(v['GHz'] for v in frames)*1000)])
 assert len(words)==672 and all(0<=v<=65535 for v in words),(len(words),max(words),min(words))
 return 'D1:'+base64.b85encode(struct.pack('<672H',*words)).decode()

def overlap_diag_unpack(blob):
 import base64,struct
 assert blob.startswith('D1:');words=list(struct.unpack('<672H',base64.b85decode(blob[3:])));out=[];i=0
 for m in (8,16,32):
  for kind in ('BASE','WS8'):
   warps=[]
   for w in range(8 if kind=='BASE' else 9):
    warps.append([words[i+k]*(32 if k<7 else 1) for k in range(10)]);i+=10
   events=[None if v==65535 else v*32 for v in words[i:i+24]];i+=24
   obs,se,mhz=words[i:i+3];i+=3;out.append(dict(M=m,kind=kind,warp_rows=warps,phase_fields=['Bfill_ns','expert_ready_ns','slot_wait_ns','dequant_MMA_ns','partial_store_barrier_ns','copy_address_issue_ns','epilogue_ns','tiles','segments','copies'],first8_issue_ns=events[:8],first8_math_begin_ns=events[8:16],first8_math_end_ns=events[16:],observer_ratio=obs/1000,observer_SE=se/10000,clock_MHz=mhz))
 return dict(records=out,quantum_ns=32,scope='Four-seed median of per-active-CTA mean per-warp residence;seed0 criticalCTA first8 issue/math events;clock32ns transport precision. Observer ratio measured against respective untraced kernel;not issue utilization,exclusive stall counters or additive critical time.')
