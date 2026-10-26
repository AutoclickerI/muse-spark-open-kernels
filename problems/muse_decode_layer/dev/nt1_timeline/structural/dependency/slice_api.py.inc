_DEP_MODULE=[]
def dep_module():
 if not _DEP_MODULE:
  target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
  mod=load_inline(name='muse_dependency_'+hashlib.sha256(DEP_CUDA.encode()).hexdigest()[:12],cpp_sources=DEP_CPP,cuda_sources=DEP_CUDA,functions=['dep_init','dep_info','dep_run'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False);mod.dep_init();_DEP_MODULE.append(mod)
 return _DEP_MODULE[0]

def dep_pack(t,S):
 # Every original weight/code and scale byte is retained once. No dequantization.
 E=t.numel()//(256*36864);NS=4096//S;GP=S//64;b=t.view(E,256,36864)
 raw=b[:,:,:32768].contiguous().view(E,256,NS,GP,32,16).permute(0,2,1,3,4,5).contiguous().view(E,NS,32,8*GP*512)
 sf=b[:,:,32768:].contiguous().view(E,256,8,32,8,2).permute(0,1,2,4,3,5).contiguous().view(E,256,NS,GP,32,2).permute(0,2,1,4,3,5).contiguous().view(E,NS,32,8*32*GP*2)
 return torch.cat((raw,sf),dim=-1).contiguous()

def dep_bind(state,S):
 dep_module();key='dep'+str(S)
 if key not in state['banks']:state['banks'][key]=dep_pack(state['banks']['lop3']['w2p'],S)
 return state

def dep_experts(data,state,S=128,mode=0,grid=264):
 x=data['hidden_states'].contiguous();M=x.shape[0];P=M*8;dev=x.device;mod=dep_module();info=mod.dep_info(S,mode);grid=min(grid,state['sms']*info[0])
 if grid<1:raise ValueError('no resident slice CTA')
 partial=torch.empty(((4096//S if mode==1 else 1),P,4096),device=dev,dtype=torch.float32)
 z=torch.empty((P,4096),device=dev,dtype=torch.float32);norm=torch.empty((P,),device=dev,dtype=torch.float32);out=torch.empty_like(x);chk=torch.empty((grid,256),device=dev,dtype=torch.float32)
 ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous()
 v=[x,ids,tw,state['pnw'],state['banks']['lop3']['w13p'],state['banks']['dep'+str(S)],state['gs13'],state['gs2'],partial,z,norm,out,chk]
 mod.dep_run([a.data_ptr() for a in v],M,data['config']['local_start'],S,mode,grid,torch.cuda.current_stream(dev).cuda_stream)
 return out
