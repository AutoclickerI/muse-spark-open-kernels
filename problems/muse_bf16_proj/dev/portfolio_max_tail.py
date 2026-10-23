class _MaxModule(_AllModule):
    def __init__(self,path):
        super().__init__(path)
        P,I=self.ct.c_void_p,self.ct.c_int
        self.lib.max_tile.argtypes=[P]*4+[I]*11+[P]
        self.lib.max_wide.argtypes=[P]*3+[I]*11+[P]

    def tile(self,w,lo,x,out,nt,tr,u,wn,ks,st,ld):
        self._check(self.lib.max_tile(w.data_ptr(),lo.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],out.shape[1],x.shape[1],
                    nt,tr,u,wn,ks,int(out.dtype==torch.float32),st,ld,torch.cuda.current_stream(x.device).cuda_stream),'shared-X tile')

    def wide(self,w,x,out,bt,ns,nc,st,ks,mc):
        self._check(self.lib.max_wide(w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],out.shape[1],x.shape[1],w.shape[1],int(out.dtype==torch.float32),
                    bt,ns,nc,st,ks,mc,torch.cuda.current_stream(x.device).cuda_stream),'multicast WGMMA')


def _dg_load():
    import deep_gemm
    return deep_gemm.bf16_gemm_nt


def _all_run(data,state,config):
    if config[0] not in ('tile','wide','deep_gemm'):return _base_run(data,state,config)
    x=data['x'];layer=data['config']['layer'];n=data['config']['n']
    out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.float32 if layer=='router' else torch.bfloat16)
    if config[0]=='tile':
        w=state['router_hi'] if layer=='router' else state[layer]
        lo=state['router_lo'] if layer=='router' else w
        state['mod'].tile(w,lo,x,out,*config[1:])
    elif config[0]=='wide':
        w=state['router_pair'] if layer=='router' else state[layer]
        state['mod'].wide(w,x,out,*config[1:])
    else:state['deep_gemm_fn'](x,state[layer][:n],out)
    return out


def _all_variants(state,layer,m,n,k,props):
    yield from _base_variants(state,layer,m,n,k,props)
    for nt,tr,u,na,st,ld in TILE_SPECS:
        if (na==2)!=(layer=='router') or nt*8>max(32,m):continue
        for wn in (4,8):
            if n%(wn*tr*16):continue
            for ks in (1,2,4,8,16):
                if (k//ks)%(32*u):continue
                shared=max(st*8*nt*4*u*16,8*min(nt,8)*(wn*tr*16+4)*4)
                if shared<=props.shared_memory_per_block_optin:yield ('tile',nt,tr,u,wn,ks,st,ld)
    if layer!='router' and m>=16:yield ('deep_gemm',)
    if _hopper():
        bt=8 if m<=8 else 16 if m<=16 else 32 if m<=32 else 128 if m<=128 else 256
        for tile,ns,nc,st,mc in WIDE_SPECS:
            if tile!=bt:continue
            br=64*ns*nc
            if n%br or (n//br)%mc:continue
            shared=max(st*(br+tile)*128+16*st+1024,64*(br+4)*4+1024)
            if shared>props.shared_memory_per_block_optin:continue
            for ks in (1,2,4,8,16):
                if ks*mc<=16:yield ('wide',tile,ns,nc,st,ks,mc)
