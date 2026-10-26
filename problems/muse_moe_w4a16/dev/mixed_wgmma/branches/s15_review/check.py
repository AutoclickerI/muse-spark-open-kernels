"""Full-bank PACK5 proof and captured current-input, block-readiness and shape transitions."""
import hashlib
import json
from pathlib import Path
import runpy
import torch
H=Path(__file__).resolve().parent
n=runpy.run_path(str(H/'s15_review_payload.py'),init_globals={'lab_cold_l2':lambda:None})
n['review_build']();w=n['dc_weights'](5353,'cuda');states=n['review_prepare'](w);proof={}
base=states['S15']['s5_hh_base'];packets=states['PACK5']['p3b_packets']
for role,packed in packets.items():
 original=base[role].contiguous().view(torch.uint8).reshape(-1,32768)
 sf=base[role+'_sf'].contiguous().reshape(-1).view(torch.int32).to(torch.int64)&0xffffffff
 sf=(((sf>>4)|(sf<<28))&0xffffffff).to(torch.int32).view(torch.uint8).reshape(-1,4096)
 count=original.shape[0];offsets=packed[:(count+1)*8].view(torch.int64);raw_total=0
 for lo in range(0,count,64):
  ix=torch.arange(lo,min(lo+64,count),device='cuda');off=offsets[ix]
  weights=packed[off[:,None]+torch.arange(32768,device='cuda')[None,:]];assert torch.equal(weights,original[ix])
  bases=packed[off[:,None]+32768+torch.arange(16,device='cuda')[None,:]]
  masks=(packed[off[:,None]+32784+torch.arange(4,device='cuda')[None,:]].long()*(1<<torch.arange(0,32,8,device='cuda'))).sum(1)
  j=torch.arange(4096,device='cuda');p=j//256;k=j%256;bits=k*5
  prev=((masks[:,None]&((1<<p)-1))[:,:,None]&(1<<torch.arange(16,device='cuda'))[None,None,:]).ne(0).sum(2)
  pos=off[:,None]+32800+p[None,:]*160+prev*96;raw=((masks[:,None]>>p)&1).bool()
  a=packed[pos+(bits//8)[None,:]].long();b=packed[pos+torch.minimum(bits//8+1,torch.tensor(159,device='cuda'))[None,:]].long()
  wide=packed[pos+torch.where(raw,k[None,:],torch.minimum(k,torch.tensor(159,device='cuda'))[None,:])].long()
  decoded=torch.where(raw,wide,bases[:,p].long()+(((a|(b<<8))>>(bits%8)[None,:])&31)).to(torch.uint8)
  assert torch.equal(decoded,sf[ix]),(role,lo)
  raw_total+=int(((masks[:,None]>>torch.arange(16,device='cuda'))&1).sum())
 proof[role]=dict(tiles=count,code_bytes=original.numel(),scale_bytes=sf.numel(),raw_packets=raw_total,coded_bytes=packed.numel(),net_byte_saving=1-packed.numel()/(original.numel()+sf.numel()),exact=True)
print('FULL BANK',proof,flush=True)
checks=0;worst=[0.,0.];bits={a:0 for a in n['REVIEW_NAMES'][1:]};cases_n=0
for m in (33,128,256,257):
 ds=n['dc_fixtures'](w,m);d0=ds[0];cases=[]
 for count in [c for c in (0,8,9,16,17,32,33,64,65,127,128,129,256,257) if c<=m]:
  ids=torch.full_like(d0['topk_ids'],-1);ids[:count,0]=5;ids[:min(count,33),1]=7;ids[:min(count,9),2]=8
  cases.append(dict(d0,topk_ids=ids))
 cases+=ds;wants=n['dc_oracles'](cases,w)
 graphs={a:n['MWGraph'](lambda d,a=a:n['review_call'](d,states,a),d0) for a in n['REVIEW_NAMES']}
 for d,want in zip(cases,wants):
  outputs={}
  for a,g in graphs.items():
   g.refresh(d)
   for _ in range(2):
    g.out.fill_(float('nan'));got=g.replay();e,r=n['op_error'](got,want);assert e<=.015 and r<=.075,(m,a,e,r)
    worst=[max(worst[0],e),max(worst[1],r)];checks+=1
   outputs[a]=got.clone()
  for a in bits:bits[a]+=int(torch.equal(outputs['S15'],outputs[a]))
  cases_n+=1
 print('PASS',m,checks,worst,flush=True);del graphs
# Arbitrary wide scale codes force the packet's raw branch through complete MoE graphs.
gen=torch.Generator(device='cuda').manual_seed(7171)
raw_base={**base}
for role in ('w13','w2'):
 sf=torch.randint(0,32,base[role+'_sf'].shape,generator=gen,device='cuda',dtype=torch.uint8)*4+32
 raw_base[role+'_sf']=sf
raw_states={**states};raw_states['S15']={**states['S15'],'s5_hh_base':raw_base}
raw_states['PACK5']={**states['PACK5'],'s5_hh_base':raw_base,'p3b_packets':n['REVIEW_NS']['PACK5']['p3b_packets'](raw_base)}
raw_checks=0
for m in (33,128,257):
 ds=n['dc_fixtures'](w,m);graphs=[n['MWGraph'](lambda d,a=a:n['review_call'](d,raw_states,a),ds[0]) for a in ('S15','PACK5')]
 for d in ds:
  for g in graphs:g.refresh(d)
  a,b=[g.replay().clone() for g in graphs];e,r=n['op_error'](b,a);assert e<=.015 and r<=.075,(m,'raw',e,r);raw_checks+=1
 del graphs
result=dict(payload_sha256=hashlib.sha256((H/'s15_review_payload.py').read_bytes()).hexdigest(),full_bank_proof=proof,checks=checks,input_cases=cases_n,bit_equal_to_S15=bits,worst_L2_row=worst,raw_scale_graph_checks=raw_checks,local_standin='Native S4 executes S3; READY W2 grid is SM/2 on sm120, SM on Hopper, because the two kernels cannot co-reside on sm120. Shared producer stores, fence/publication/acquire and changed math all execute.')
(H/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
