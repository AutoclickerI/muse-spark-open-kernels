"""Generate fused V9 with on-device expert-rank scheduling and bounded N chunks."""
from pathlib import Path
from build_moe_v9_reuse import raw_v9,wide_v9,ROOT


def build():
    s=wide_v9(raw_v9((ROOT/'problems/muse_moe_w4a16/cand_v9.py').read_text()))
    s=s.replace('for (int j = 0; j < NT; ++j) mma16816(acc[j],', 'for (int j = 0; j < NT; ++j) if(NT==1 || nb*NB+j*8<ne) mma16816(acc[j],')
    s=s.replace('struct Route {','struct Route {\n  int order[32], rank[32], active_grid;')
    s=s.replace('const int G = gridDim.x, c = blockIdx.x;','int G=gridDim.x; const int c=blockIdx.x;')
    needle='''  __syncthreads();
  if (warp == 0) {
    const int n = rt->n[lane];'''
    replacement='''  __syncthreads();
  if(warp==0){int n=rt->n[lane],rank=0;
    if(sched){
      #pragma unroll
      for(int e=0;e<32;++e){int other=__shfl_sync(~0u,n,e);rank+=(other>n || (other==n && e<lane));}
    }else rank=lane;
    rt->order[rank]=lane;rt->rank[lane]=rank;rt->n[rank]=n;
  }
  __syncthreads();
  for(int i=tid;i<npairs;i+=NTHR)if(eflat[i]>=0)eflat[i]=rt->rank[(int)eflat[i]];
  __syncthreads();
  if (warp == 0) {
    const int n = rt->n[lane];'''
    assert needle in s;s=s.replace(needle,replacement)
    s=s.replace('rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));',
'''rt->active_grid=min(G,max(1,(U2+7)/8));
      int cap=chunk_cap>0?chunk_cap:NT1,active=rt->active_grid;
      rt->c1 = min(cap,min(NT1, max(1, (U1 + active / 2) / active)));
      rt->c2 = min(cap,min(NT2, max(1, (U2 + active / 2) / active)));''')
    s=s.replace('  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];','  G=rt->active_grid; if(c>=G)return;\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];')
    s=s.replace('(static_cast<size_t>(e)*ntile+m)*TILE_BYTES','(static_cast<size_t>(rt->order[e])*ntile+m)*TILE_BYTES')
    s=s.replace('gs13[e] * 256.f','gs13[rt->order[e]] * 256.f').replace('gs2[e] * 256.f','gs2[rt->order[e]] * 256.f')
    s=s.replace('int M, int local_start, int nstage) {','int M, int local_start, int nstage,int sched,int chunk_cap) {')
    s=s.replace('int64_t grid) {','int64_t grid,int64_t sched,int64_t chunk_cap) {').replace('int64_t grid);','int64_t grid,int64_t sched,int64_t chunk_cap);')
    s=s.replace('static_cast<int>(nstage))','static_cast<int>(nstage),static_cast<int>(sched),static_cast<int>(chunk_cap))')
    s=s.replace('state["grid"] * mod.resident_blocks(nt, nstage, M))','state["grid"] * mod.resident_blocks(nt, nstage, M),state.get("sched",1),state.get("chunk_cap",0))')
    # Amortize normal E4M3 conversion across four g groups. Finite negative/subnormal cases
    # use the same exact byte decoder; generated weights are positive normals.
    old="""const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};"""
    new="""const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};
        uint32_t sf0[8],sf1[8];
        #pragma unroll
        for(int g=0;g<8;++g){uint32_t sw=scw[g>>1];sf0[g]=sf_bits((sw>>((g&1)*16))&255u);sf1[g]=sf_bits((sw>>((g&1)*16+8))&255u);}"""
    s=s.replace(old,new).replace('const uint32_t Sr = sf_bits((sw >> ((g & 1)*16)) & 255u);','const uint32_t Sr=sf0[g];').replace('const uint32_t Sr8 = sf_bits((sw >> ((g & 1)*16+8)) & 255u);','const uint32_t Sr8=sf1[g];')
    s=s.replace('nstage = 2\n','nstage = state.get("stages",2)\n')
    p=ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/hot_v9.py';p.write_text(s)
    return p


if __name__=='__main__':print(build())
