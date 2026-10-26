# Own host wrapper; the donor autotuner/diagnostics/driver are not imported.
def prepare_lee(weights):
 dev=weights['w13'].device;E=32;I=4096
 perm1=torch.stack((torch.arange(I,device=dev).view(-1,8),I+torch.arange(I,device=dev).view(-1,8)),1).reshape(-1)
 perm2=torch.arange(I,device=dev)
 w1=torch.stack([_repack(weights['w13'][e],weights['w13_sf'][e],perm1) for e in range(E)])
 w2=torch.stack([_repack(weights['w2'][e],weights['w2_sf'][e],perm2) for e in range(E)])
 w2f=torch.stack([_repack2(weights['w2'][e],weights['w2_sf'][e]) for e in range(E)])
 return dict(w1=w1,w2f=w2f,w1x=_repackX(w1),w2x=_repackX(w2),g1=weights['w13_gs'][:E].float()*256.,g2=weights['w2_gs'][:E].float()*256.,nw=1.+weights['post_norm_w'].float())

def lee_plan(data,state,mod,kind='X',split=False):
 x=data['hidden_states'];M=x.shape[0];dev=x.device;ids=data['topk_ids'];tw=data['topk_weights'];lo=data['config']['local_start'];out=torch.empty_like(x)
 if kind=='F':
  yacc=torch.zeros((M*8,4096),device=dev);ctr=torch.zeros(9280+64,dtype=torch.int32,device=dev);prof=torch.empty(0,dtype=torch.int64,device=dev)
  ts=[x,ids,tw,state['w1'],state['w2f'],state['g1'],state['g2'],state['nw'],yacc,ctr,out,prof];ptr=[t.data_ptr() for t in ts]
  def fused():mod.port_f(ptr,M,lo,torch.cuda.current_stream().cuda_stream)
  steps=[('fused',fused)]
 else:
  rows=M*8+32*7;xg=torch.empty(rows*8192,dtype=torch.uint8,device=dev);ig=torch.empty_like(xg);y=torch.empty((M*8,4096),dtype=torch.bfloat16,device=dev);ss=torch.empty(M*8,device=dev);ctr=torch.zeros(64,dtype=torch.int32,device=dev);prof=torch.empty(0,dtype=torch.int64,device=dev)
  ts=[x,ids,tw,state['w1x'],state['w2x'],state['g1'],state['g2'],state['nw'],xg,ig,y,ss,ctr,out,prof];ptr=[t.data_ptr() for t in ts]
  def call(mode):mod.port_x(ptr,M,lo,mode,torch.cuda.current_stream().cuda_stream)
  steps=[('route',lambda:call(0))]+([('w13',lambda:call(2)),('w2',lambda:call(3)),('finish',lambda:call(4))] if split else [('fused',lambda:call(1))])
 return dict(steps=steps,out=out,owners=ts)
