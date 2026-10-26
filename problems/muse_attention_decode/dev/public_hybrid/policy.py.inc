PICKS={'16x1s':'L','16x1f':'O','32x1s':'L','1x8s':'O','1x8f':'L','16x8s':'L','16x8f':'L','32x8s':'L'}
def cell(data):
 c=data['config'];return f"{c['batch']}x{c['q_len']}"+('s' if c['window'] else 'f')
def covered(data):
 c=data['config'];return c['num_q_heads']==16 and c['num_kv_heads']==2 and c['head_dim']==64 and c['page_size']==1 and c['batch'] in (1,16,32) and c['q_len'] in (1,8)
def arms(data):
 if not covered(data):return ('A','A2','F')
 c=data['config'];b,l,w=c['batch'],c['q_len'],c['window']
 if l==1 and b==1 and w:return ('A','A2','F','J')
 if l==1:
  return ('A','A2','F') if b==1 else ('A','A2','F','O') if w else ('A','A2','F','K')
 if w:
  return ('A','A2','F') if b==1 else ('A','A2','F','O','P','E') if b==16 else ('A','A2','P','E')
 return ('A','A2','F','P','Q','E','H')
def call(name,pair,data):
 c=data['config'];b,l,w=c['batch'],c['q_len'],c['window']
 if name in ('A','A2'):
  pick=PICKS.get(cell(data),'A') if covered(data) else 'A'
  if pick=='A':return base_call(pair,data)
  return (op if pick=='O' else lee).attention(pack.raw(pair,data),data)
 if not covered(data):return base_call(pair,data)
 if w and (name=='O' or b==1 and name!='J'):return op_direct.attention_pair(pair,data)
 return lee_hybrid.attention_pair(pair,data,name)
