def attention_pair(pair,data):
 p,sp=pair;c=data['config'];B,L=c['batch'],c['q_len'];sl=bool(c['window']);dev=p.device
 setup(dev);sms=torch.cuda.get_device_properties(dev).multi_processor_count
 span=min(data['page_table'].shape[1],c['window']+L) if sl else data['page_table'].shape[1]
 nl=1 if _LOCAL else 2
 if L==1:
  assert sl and B in (16,32)
  kind=1;ns=max(nl,min((span+63)//64,nl*(sms//B)));ns-=ns%nl;x=64;cap=2
 else:
  assert (B,sl) in VCFG
  var,kps,cap,x=VCFG[B,sl];kind=3
  ns=max(1,min((span+kps-1)//kps,264//B))
  ns=max(nl,min(ns+(ns&1) if nl==2 else ns,nl*(sms//B)))
  if x&256:ns=min(64,2*((264+B-1)//B))
 key=(kind,B,L,ns)
 if key not in _BUF:
  rows=16 if L==1 else 128
  _BUF[key]=(torch.empty(B*ns*rows*64,device=dev),torch.empty(B*ns*rows*2,device=dev),torch.zeros(2*B,device=dev,dtype=torch.int32))
 wo,wml,cnt=_BUF[key];out=torch.empty((B,L,1024),device=dev,dtype=torch.bfloat16)
 _MOD.lee_attention(p,data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],wo,wml,cnt,out,_INV[str(dev),float(c['rope_theta'])],_EMPTY,kind,c['window'],ns,x,c['rope_theta'],c['qk_scale_factor'],c['sm_scale'],c['kv_scale'],cap)
 return out
