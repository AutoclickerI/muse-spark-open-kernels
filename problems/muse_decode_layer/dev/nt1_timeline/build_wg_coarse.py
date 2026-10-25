"""K128/K256 WGMMA pipelines, input permutation fused into producer stores, vector activation copies."""
import ast,hashlib
from pathlib import Path
from build_wg_perm import source as perm_source
DEV=Path(__file__).resolve().parent

def bundled_mma():
 cases=[]
 for n in (8,16,32):
  outputs=n//2;ctuple='{'+','.join('%'+str(i) for i in range(outputs))+'}'
  desc='%'+str(outputs+16)
  asm='{ .reg .pred p; setp.ne.b32 p,1,0; .reg .b64 d1,d2,d3; '
  for j in (1,2,3):asm+=f'add.u64 d{j},{desc},{j*2}; '
  asm+='wgmma.fence.sync.aligned; '
  for j in range(4):
   atuple='{'+','.join('%'+str(outputs+j*4+i) for i in range(4))+'}'
   d=desc if j==0 else f'd{j}'
   asm+=f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {ctuple},{atuple},{d},p,1,1,0; '
  asm+='wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 1; }'
  outs=','.join(f'"+f"(c[{i}])' for i in range(outputs));ins=','.join(f'"r"(a[{i}])' for i in range(16))+',"l"(d)'
  cases.append(('if' if not cases else 'else if')+f' constexpr(NT=={n})'+'{asm volatile("'+asm+'":'+outs+':'+ins+':"memory");}')
 return 'template<int NT>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\n#if BR_HOPPER\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n'+''.join(cases)+'\n#else\n#pragma unroll\n for(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\n#endif\n}\n'

def source(chunks):
 body,cuda=perm_source();old=cuda
 a=cuda.index('template<int NT>__device__ __forceinline__ void mma_group(');b=cuda.index('__device__ __forceinline__ void wgp_keep(',a)
 cuda=cuda[:a]+bundled_mma()+cuda[b:]
 cuda=cuda.replace('constexpr int BK=64;',f'constexpr int BK=64,CHUNKS={chunks};')
 cuda=cuda.replace('S*(2048+512+32*64*2)','S*(CHUNKS*2048+CHUNKS*256+32*64*CHUNKS*2)')
 cuda=cuda.replace('sc=qs+S*2048','sc=qs+S*CHUNKS*2048').replace('sc+S*512','sc+S*CHUNKS*256').replace('bs+S*32*64','bs+S*CHUNKS*32*64')
 cuda=cuda.replace('else asm volatile("cp.async.ca.shared.global [%0],[%1],4,%2;', 'else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8,%2;"::"r"(dst),"l"(s),"r"(valid?8:0):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4,%2;')
 a=cuda.index('   auto issue=[&]');b=cuda.index('\n#pragma unroll\n   for(int st=',a)
 cuda=cuda[:a]+'''   auto issue=[&](int slot,int ki){
#pragma unroll
    for(int sub=0;sub<CHUNKS;++sub){
     wg_cp(qs+(slot*CHUNKS+sub)*2048+t*16,weights+(ki*CHUNKS+sub)*512+l*16,16,1);
#pragma unroll
     for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;
      if(r<NT)wg_cp(bs+(slot*CHUNKS+sub)*32*64+r*64+(k^((r&7)*8)),xr[j]+(ki*CHUNKS+sub)*64+k,16,xv[j]);}
    }
    wg_cp(sc+slot*CHUNKS*256+t*CHUNKS*2,scales+(ki*32+l)*CHUNKS,CHUNKS*2,1);
    asm volatile("cp.async.commit_group;":::"memory");
   };'''+cuda[b:]
 a=cuda.index('#pragma unroll 1\n   for(int kbase=');b=cuda.index('\n#if BR_HOPPER\n   asm volatile("wgmma.wait_group.sync.aligned 0;',a)
 cuda=cuda[:a]+'''#pragma unroll 1
   for(int ki=0;ki<64/CHUNKS;++ki){int st=ki%S;
    asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");__syncthreads();
#if BR_HOPPER
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
#pragma unroll 1
    for(int pair=0;pair<CHUNKS;pair+=2){
#pragma unroll
    for(int half=0;half<2;++half){int sub=pair+half;
     uint4 q=*reinterpret_cast<uint4*>(qs+(st*CHUNKS+sub)*2048+t*16);
     uint32_t sw=*reinterpret_cast<uint32_t*>(sc+st*CHUNKS*256+t*CHUNKS*2+(sub>>1)*4);sw>>=(sub&1)*16;
     uint32_t sr=__byte_perm(sw,0u,0x4040u)*16u+0x70007000u,sr8=__byte_perm(sw,0u,0x4141u)*16u+0x70007000u;
     uint32_t o[16];dq8(q.x,q.y,o);dq8(q.z,q.w,o+8);unsigned* ar=af[sub&1];
#pragma unroll
     for(int z=0;z<4;++z){int k=(z>>1)*8+(z&1)*4;ar[z*4]=hmul2(o[k],sr);ar[z*4+1]=hmul2(o[k+1],sr8);ar[z*4+2]=hmul2(o[k+2],sr);ar[z*4+3]=hmul2(o[k+3],sr8);}
     wg_keep(ar);
#if BR_HOPPER
     asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
     mma_group<NT>(acc,ar,bs+(st*CHUNKS+sub)*32*64);wg_keep(af[(sub&1)^1]);
     if(sub==0){if(ki+S-1<64/CHUNKS)issue((ki+S-1)%S,ki+S-1);else asm volatile("cp.async.commit_group;":::"memory");}
    }}
   }'''+cuda[b:]
 # perm_source already renamed helper entrypoints; new snippets use the original short prefix.
 cuda=cuda.replace('#if BR_HOPPER\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n', '')
 cuda=cuda.replace('unsigned af[2][16]', 'unsigned af[2][16]')
 cuda=cuda.replace('ar=af[sub&1]', 'ar=af[half]')
 cuda=cuda.replace('wg_keep(af[(sub&1)^1]);', '\n#pragma unroll\n     for(int keep=0;keep<2;++keep)wg_keep(af[keep]);')
 cuda=cuda.replace('wg_keep(af[0]);wg_keep(af[1]);', '\n#pragma unroll\n   for(int keep=0;keep<2;++keep)wg_keep(af[keep]);')
 cuda=cuda.replace('wg_','wgp_')
 if chunks==4:cuda=cuda.replace('ATTR(4) ATTR(8)','ATTR(2) ATTR(4)').replace('INFO(4) INFO(8)','INFO(2) INFO(4)').replace('LAUNCH(4) LAUNCH(8)','LAUNCH(2) LAUNCH(4)')
 body=body.replace(repr(old),repr(cuda),1)
 body=body.replace('.view(32,ntile,8,32,8,2)',f'.view(32,ntile,8,32,{8//chunks},{chunks*2})')
 if chunks==4:body=body.replace('for s in (4,8)','for s in (2,4)').replace('stages=8','stages=4')
 prefix=f'wg{chunks}_';upper=f'WG{chunks}_'
 body=body.replace('wgp_',prefix).replace('WGP_',upper)
 cuda=cuda.replace('wgp_',prefix).replace('WGP_',upper)
 return body,cuda

def build():
 for c in (2,4):
  body,cuda=source(c)
  for p,s in ((DEV/f'wg{c}_cell.py',body),(DEV/f'wg{c}.cu',cuda)):
   p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
