def ext(clocks=False):return vq8_ext(clocks)
def attention(data,clocks=None,plan=None,qkv=None):
 cfg=data['config'];assert cfg['q_len']==1 and cfg['window']==0
 raw=data['qkv'] if qkv is None else qkv
 kw={}
 if isinstance(raw,tuple):
  part,splits=raw
  if part.dtype==torch.bfloat16:raw=part[0] if part.dim()==3 else part
  else:raw=part;kw=dict(proj_splits=splits,split_stride=part.stride(0))
 return vq8_attention(data,qkv=raw,stages=6,min_tiles=8,clocks=clocks,**kw)
