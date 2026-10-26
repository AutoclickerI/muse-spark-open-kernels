"""Amortize WGMMA synchronization over K128, with four live A register sets."""
def da_grouped_source(ns,instances):
    src=da_wide_source(ns,instances,False)
    for statement in ('wgmma.fence.sync.aligned;','wgmma.commit_group.sync.aligned;','wgmma.wait_group.sync.aligned 0;'):src=src.replace(statement,'')
    begin=src.index('  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {');end=src.index('  __syncthreads();\n  if(tid%128==0)',begin)
    decode_begin=src.index('      if constexpr(C12)',begin);decode_end=src.index('      int s=i%Q;',decode_begin);decode=src[decode_begin:decode_end]
    loop='''  uint32_t abuf[2][2][16]={};
  #pragma unroll 1
  for(int outer=0;outer<nkb;outer+=4) {
    #pragma unroll
    for(int slot=0;slot<2;++slot) {
    int ib=outer+2*slot;if(ib>=nkb)break;
    #pragma unroll
    for(int p=0;p<2;++p) {
      int i=ib+p;uint32_t* a=abuf[slot][p];
'''+decode+'''
      waitbar(full+(i%Q)*8,(i/Q)&1);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
    #pragma unroll
    for(int p=0;p<2;++p)for(int j=0;j<16;++j)asm volatile(""::"r"(abuf[1-slot][p][j]):"memory");
    if(ib>0){loadx(ib+Q-2);loadx(ib+Q-1);}
    #pragma unroll
    for(int j=0;j<BN/2;++j)asm volatile("":"+f"(acc[j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    #pragma unroll
    for(int p=0;p<2;++p) {
      uint32_t* a=abuf[slot][p];uint64_t d=desc(base+((ib+p)%Q)*SB);
      if constexpr(BN==64)mma64(acc,a,d);
      if constexpr(BN==128)mma128(acc,a,d);
    }
    asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
  #pragma unroll
  for(int b=0;b<2;++b)for(int p=0;p<2;++p)for(int j=0;j<16;++j)asm volatile(""::"r"(abuf[b][p][j]):"memory");
  #pragma unroll
  for(int j=0;j<BN/2;++j)asm volatile("":"+f"(acc[j])::"memory");
'''
    src=src[:begin]+loop+src[end:]
    # Adjacent token tiles read the same weights back-to-back. This matters
    # when a pool bank exceeds L2 and BN128 covers a T256 call in two tiles.
    src=src.replace('int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;','int rank=blockIdx.y%split,nt=blockIdx.y/split,mt=blockIdx.x;')
    src=src.replace('cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);','cfg.gridDim=dim3((m+BN-1)/BN,(n/R)*split);')
    return src.replace('p10_daw_normal','p10_daw_grouped')
