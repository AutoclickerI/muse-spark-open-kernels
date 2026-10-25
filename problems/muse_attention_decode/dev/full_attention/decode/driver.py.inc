def ext(clocks=False):
 key=bool(clocks)
 if key not in _EXT:
  major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
  tag=hashlib.sha256(CUDA.encode()).hexdigest()[:12]
  _EXT[key]=load_inline(name=f'decode_soft_{tag}_{arch}_{int(key)}',cpp_sources=CPP,cuda_sources=CUDA,
   functions=['attn_forward','max_cluster','occupancy','decode_probes'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',
    f'-gencode=arch=compute_{arch},code=sm_{arch}',f'-DDECODE_CLOCK={int(key)}','-lineinfo'],verbose=False)
 return _EXT[key]

def plan_for(data):
 cfg=data['config'];dev=data['qkv'].device
 key=(str(dev),cfg['batch'],cfg['num_kv_heads'],cfg['page_size'],data['page_table'].shape[1])
 if key not in _PLANS:
  e=ext();sms=torch.cuda.get_device_properties(dev).multi_processor_count
  occ=(max(1,e.occupancy(1)),{cs:e.max_cluster(1,cs) for cs in range(1,17)})
  _PLANS[key]=_attn_main_plan(cfg['batch'],cfg['num_kv_heads']//2,key[3]*key[4],1,0,sms,occ)
 return _PLANS[key]

def attention(data,clocks=None,plan=None,qkv=None):
 cfg=data['config'];B,HK,HQ=cfg['batch'],cfg['num_kv_heads'],cfg['num_q_heads']
 assert cfg['q_len']==1 and cfg['window']==0 and HQ==8*HK
 raw=data['qkv'] if qkv is None else qkv
 if isinstance(raw,tuple):
  part,splits=raw
  raw=(part[0] if part.dim()==3 else part) if part.dtype==torch.bfloat16 else part.sum(0).to(torch.bfloat16)
 dev=raw.device;e=ext(clocks is not None);ns,cs=plan or plan_for(data)
 rows=B*(HK//2)*(ns//cs)*16;po=torch.empty(rows*64,device=dev,dtype=torch.float32)
 nclock=B*(HK//2)*ns*4*32*2 if clocks is not None else 0
 ml=torch.empty(rows*2+nclock,device=dev,dtype=torch.float32)
 key=(str(dev),B,HK,cs)
 if key not in _COUNTERS:_COUNTERS[key]=torch.zeros(B*(HK//2)*cs,device=dev,dtype=torch.int32)
 out=torch.empty((B,1,HQ*64),device=dev,dtype=torch.bfloat16)
 e.attn_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),data['seq_lens'].data_ptr(),
  data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),0,po.data_ptr(),ml.data_ptr(),
  _COUNTERS[key].data_ptr(),po.data_ptr(),1,out.data_ptr(),B,1,raw.stride(-2),HQ,HK,data['page_table'].shape[1],cfg['page_size'],0,ns,cs,
  cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,torch.cuda.current_stream(dev).cuda_stream)
 if clocks is not None:clocks.append(ml[rows*2:].view(torch.int64).view(-1,32))
 return out
