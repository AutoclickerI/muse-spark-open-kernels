"""Overlap decoding the next register-A tile with the current WGMMA group."""
def da_pipeline_source(ns,instances,pipeline=True):
    src=da_wide_source(ns,instances,False)
    if pipeline:
        # Two A register sets stay live through their matching wait. X has Q=2
        # shared slots; replenish a slot only after its previous MMA has drained.
        src=src.replace('wgmma.wait_group.sync.aligned 0;','')
        old='  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {'
        assert src.count(old)==1
        src=src.replace(old,'  uint32_t abuf[2][16]={};\n'+old)
        src=src.replace('      uint32_t a[16];','      uint32_t* a=abuf[p];')
        old='      int s=i%Q;\n      waitbar(full+s*8,(i/Q)&1);'
        new='''      int s=i%Q;
      waitbar(full+s*8,(i/Q)&1);
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      // Previous A stays live while this tile decodes, so the register
      // allocator cannot alias the two asynchronous operand sets.
      #pragma unroll
      for(int j=0;j<16;++j)asm volatile(""::"r"(abuf[1-p][j]):"memory");
      if(i>0)loadx(i+Q-1);'''
        assert src.count(old)==1;src=src.replace(old,new)
        old='''      #pragma unroll
      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");
      loadx(i+Q);'''
        assert src.count(old)==1;src=src.replace(old,'')
        old='''  }
  __syncthreads();
  if(tid%128==0) {'''
        new='''  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  #pragma unroll
  for(int b=0;b<2;++b)for(int j=0;j<16;++j)asm volatile(""::"r"(abuf[b][j]):"memory");
  #pragma unroll
  for(int j=0;j<BN/2;++j)asm volatile("":"+f"(acc[j])::"memory");
  __syncthreads();
  if(tid%128==0) {'''
        assert src.count(old)==1;src=src.replace(old,new)
    tag='p10_daw_pipeline' if pipeline else 'p10_daw_oracle'
    return src.replace('p10_daw_normal',tag)
