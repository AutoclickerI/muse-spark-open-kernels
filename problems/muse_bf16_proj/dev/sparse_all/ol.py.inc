def dc_ol_source(src):
    """Decode O weights directly into the existing SS-WGMMA shared ring (O32)."""
    src=DC_NATIVE+'\n'+src
    a=src.index('    asm volatile("mbarrier.arrive.expect_tx',src.index('  auto load='))
    b=src.index('\n  };',a)
    src=src[:a]+'''    static_assert(IT==ST && !(PIPE&2) && TW==1 && PACK==1);
    for(int q=tid;q<64*BK;q+=TH) {
      int row=q/BK,col=q%BK;
      __nv_bfloat16 v=dc_point(w,nt*64+row,i*BK+col);
      int off=(col/64)*8192+row*128+((col%64)^((row&7)*8))*2;
      *(__nv_bfloat16*)(sm+s*SB+off)=v;
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
    if(tid==0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(full+s*8),"r"(T*BK*2):"memory");
      r3_o32_o_direct_a_ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
    }
'''+src[b:]
    old='''    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();'''
    assert old in src
    src=src.replace(old,'''  }
  __syncthreads();
  #pragma unroll
  for(int i=0;i<ST && i<IT;i++) load(i);
  __syncthreads();''')
    # No compressed pointer is described to TMA; only activations still use TMA.
    old='CUtensorMap tw,tx;\n  int rc=r3_o32_o_direct_a_ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;\n  rc=r3_o32_o_direct_a_ol_map(&tx'
    assert old in src
    src=src.replace(old,'CUtensorMap tw{},tx;\n  int rc=r3_o32_o_direct_a_ol_map(&tx')
    return src


def dc_prepare_ol(st):
    st['ol']=None
    if st['native']:
        ns=_dc_base_nf_vendor();ns['b2o_OLAYER_CUDA']=_nf_pdl_wide(dc_ol_source(ns['b2o_OLAYER_CUDA']))
        mod=ns['b2o_build_olayer']([(32,128,8,1,1,1)])
        st['ol']=mod


def dc_ol_project(x,st):
    c=st['sparse']['o'];out=torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
    if st['native']:
        rc=st['ol'].r3_o32_o_direct_a_ol_launch(0,x.data_ptr(),c['coded'].data_ptr(),out.data_ptr(),x.shape[0],torch.cuda.current_stream().cuda_stream,0)
        if rc:raise RuntimeError('sparse O32 WGMMA '+str(rc))
    else:
        # Actual bank decoding and O32 allocation/output ABI, WGMMA launch stand-in.
        w=torch.empty((8192,1024),device=x.device,dtype=torch.bfloat16);dc_unpack(c,w);out.copy_(torch.mm(x,w.T))
    return out.unsqueeze(0),1
