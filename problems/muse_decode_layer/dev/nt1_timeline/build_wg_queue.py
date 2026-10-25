"""Independent deep raw-weight TMA queue and three activation stages, exact packed weights."""
import hashlib
from pathlib import Path
from build_wg_coarse import source as coarse_source
DEV=Path(__file__).resolve().parent

def source():
 body,cuda=coarse_source(2);old=cuda
 a=cuda.index('template<int S>constexpr int wg2_smem');b=cuda.index('__device__ __forceinline__ void wg2_cp',a)
 cuda=cuda[:a]+'''template<int S>constexpr int wg2_smem(int M){return S*4608+3*32*128*2+S*8+sizeof(WGRoute)+((M*8+15)&~15)+4*M*8+4*32*4;}
'''+cuda[b:]
 a=cuda.index(' auto* qs=sm;');b=cuda.index(' auto* ef=',a) if ' auto* ef=' in cuda[a:] else -1
 # rt and ef share one source line.
 end=cuda.index('auto* ef=',a)
 cuda=cuda[:a]+''' auto* qs=sm;auto* bs=reinterpret_cast<__nv_bfloat16*>(qs+S*4608);
 auto* full=reinterpret_cast<uint64_t*>(bs+3*32*128);auto* rt=reinterpret_cast<WGRoute*>(full+S);'''+cuda[end:]
 a=cuda.index('   const uint8_t* weights=');b=cuda.index('   constexpr int NJ=',a)
 cuda=cuda[:a]+'''   const uint8_t* weights=(pass?w2:w13)+(static_cast<size_t>(e)*(ntile/4)+mt)*32*4608;
'''+cuda[b:]
 a=cuda.index('   auto issue=[&]');b=cuda.index('   float acc[NT/2]',a)
 cuda=cuda[:a]+'''   if(t==0){
#pragma unroll
    for(int st=0;st<S;++st)mbar_init(full+st,1);
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
   }
   __syncthreads();
   uint64_t policy=0;if(t==0)asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(policy));
   auto weight_issue=[&](int ki){if(t==0){int slot=ki%S;mbar_expect_tx(full+slot,4608);bulk_g2s(qs+slot*4608,weights+ki*4608,4608,full+slot,policy);}};
   auto act_issue=[&](int ki){int slot=ki%3;
#pragma unroll
    for(int sub=0;sub<2;++sub){
#pragma unroll
     for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;
      if(r<NT)wg2_cp(bs+(slot*2+sub)*32*64+r*64+(k^((r&7)*8)),xr[j]+(ki*2+sub)*64+k,16,xv[j]);}
    }
    asm volatile("cp.async.commit_group;":::"memory");
   };
#pragma unroll
   for(int st=0;st<S;++st)weight_issue(st);
   act_issue(0);act_issue(1);
'''+cuda[b:]
 a=cuda.index('#pragma unroll 1\n   for(int ki=');b=cuda.index('\n#if BR_HOPPER\n   asm volatile("wgmma.wait_group.sync.aligned 0;',a)
 cuda=cuda[:a]+'''#pragma unroll 1
   for(int ki=0;ki<32;++ki){int st=ki%S,as=ki%3;
    if(t==0)mbar_wait(full+st,(ki/S)&1);
    asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();
#if BR_HOPPER
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
#pragma unroll
    for(int half=0;half<2;++half){
     uint4 q=*reinterpret_cast<uint4*>(qs+st*4608+half*2048+t*16);
     uint32_t sw=*reinterpret_cast<uint32_t*>(qs+st*4608+4096+half*256+(t&~1)*2);sw>>=(t&1)*16;
     uint32_t sr=__byte_perm(sw,0u,0x4040u)*16u+0x70007000u,sr8=__byte_perm(sw,0u,0x4141u)*16u+0x70007000u;
     uint32_t o[16];dq8(q.x,q.y,o);dq8(q.z,q.w,o+8);unsigned* ar=af[half];
#pragma unroll
     for(int z=0;z<4;++z){int k=(z>>1)*8+(z&1)*4;ar[z*4]=hmul2(o[k],sr);ar[z*4+1]=hmul2(o[k+1],sr8);ar[z*4+2]=hmul2(o[k+2],sr);ar[z*4+3]=hmul2(o[k+3],sr8);}
     wg2_keep(ar);mma_group<NT>(acc,ar,bs+(as*2+half)*32*64);
#pragma unroll
     for(int keep=0;keep<2;++keep)wg2_keep(af[keep]);
    }
    // The group for ki-1 has retired after this pair. Its activation slot can be reused for ki+2.
    __syncthreads();if(ki+S<32)weight_issue(ki+S);
    if(ki+2<32)act_issue(ki+2);else asm volatile("cp.async.commit_group;":::"memory");
   }
   asm volatile("cp.async.wait_group 0;":::"memory");
'''+cuda[b:]
 cuda=cuda.replace('ATTR(4) ATTR(8)','ATTR(4) ATTR(8) ATTR(12)').replace('INFO(4) INFO(8)','INFO(4) INFO(8) INFO(12)').replace('LAUNCH(4) LAUNCH(8)','LAUNCH(4) LAUNCH(8) LAUNCH(12)')
 body=body.replace(repr(old),repr(cuda),1)
 a=body.index(' scales={}');b=body.index(' return state',a)
 body=body[:a]+''' packed={}
 for name in ('w13p','w2p'):
  src=bank[name];ntile=src.shape[1];mt=ntile//4
  q=src[:,:,:32768].contiguous().view(32,mt,4,64,32,16).permute(0,1,3,2,4,5).contiguous().view(32,mt,32,4096)
  sc=src[:,:,32768:].contiguous().view(32,ntile,8,32,8,2).permute(0,1,2,4,3,5).contiguous()
  sc=sc.view(32,mt,4,64,32,2).permute(0,1,3,2,4,5).contiguous().view(32,mt,32,512)
  packed[name]=torch.cat((q,sc),dim=-1).contiguous()
 state['wg2_packed']=packed;state['wg2_counters']={(s,g):torch.zeros(64,device=dev,dtype=torch.int32) for s in (4,8,12) for g in (2,4)}
'''+body[b:]
 body=body.replace("sf=state['wg2_scales']", "sf=state['wg2_packed']")
 body=body.replace("bank['w13p'],bank['w2p'],sf['w13p'],sf['w2p']", "sf['w13p'],sf['w2p'],sf['w13p'],sf['w2p']")
 body=body.replace('wg2_','wgq_').replace('WG2_','WGQ_');cuda=cuda.replace('wg2_','wgq_').replace('WG2_','WGQ_')
 return body,cuda

def build():
 body,cuda=source()
 for p,s in ((DEV/'wgq_cell.py',body),(DEV/'wgq.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
