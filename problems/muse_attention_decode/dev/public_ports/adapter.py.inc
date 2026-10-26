# Original S19 dispatch (including R8 and window/R2 rows); owned input packing
# is timed before the selected reviewed donor entry, never in prepare.
_PORT_STATE={}
def base_call(pair,data):
 c=data['config'];b,l=c['batch'],c['q_len'];d=dict(data,config=dict(c,tokens=b*l))
 if c['num_q_heads']!=16:return base.attention_partial_port(pair,d,base._PP_FAST).reshape(b,l,-1)
 key=str(pair[0].device),c['rope_theta']
 if key not in _PORT_STATE:
  _PORT_STATE[key]=dict(inv_freq=base._inv_freq(c['rope_theta'],pair[0].device),cluster_rows=base._CLUSTER_ROWS,partial_port_rows=base._PP_ROWS,vq8_rows=base._VQ8_ROWS,attention_plan={},r2_rows=base._R2_SELECTION,wr_rows=base._WINDOW_SELECTION)
 setting=base.BEST7_SETTINGS.get((b,l,c['window']),base.BEST7_SETTINGS['other']);glue=setting[0][0] if isinstance(setting[0],tuple) else setting[0]
 return base._best7_attention(pair,d,_PORT_STATE[key],dict(glue=glue,attention=setting[1])).reshape(b,l,-1)
def covered_lee(data):
 c=data['config'];return c['num_q_heads']==16 and c['num_kv_heads']==2 and c['head_dim']==64 and c['page_size']==1 and c['batch'] in (1,16,32) and c['q_len'] in (1,8)
def call(name,pair,data):
 if name in ('A','A2') or (name=='L' and not covered_lee(data)):return base_call(pair,data)
 return (op if name=='O' else lee).attention(pack.raw(pair,data),data)
