def attention_pair(pair,data,arm='STAGE'):
 p,sp=pair;c=data['config'];B,L=c['batch'],c['q_len'];sl=bool(c['window']);assert L==8
 setup(p.device);sms=torch.cuda.get_device_properties(p.device).multi_processor_count
 span=min(data['page_table'].shape[1],c['window']+L) if sl else data['page_table'].shape[1]
 _,kps,cap,x=VCFG[B,sl];x&=~32768
 ns=max(1,min((span+kps-1)//kps,264//B));ns=max(2,min(ns+(ns&1),2*(sms//B)))
 if x&256:ns=min(64,2*((264+B-1)//B))
 mode={'STAGE':3,'LANE1':4,'SOFTMAX':5}[arm]
 key=(B,L,ns)
 if key not in _BUF:_BUF[key]=(torch.empty(B*ns*128*64,device=p.device),torch.empty(B*ns*128*2,device=p.device),torch.zeros(2*B,device=p.device,dtype=torch.int32))
 wo,wml,cnt=_BUF[key];out=torch.empty((B,L,1024),device=p.device,dtype=torch.bfloat16)
 _MOD.lee_attention(p,data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],wo,wml,cnt,out,_INV[str(p.device),float(c['rope_theta'])],_EMPTY,mode,c['window'],ns,x,c['rope_theta'],c['qk_scale_factor'],c['sm_scale'],c['kv_scale'],cap)
 return out
