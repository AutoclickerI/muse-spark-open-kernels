"""Resident sliding history: one KV pass, in-place V transpose, early PDL staging and pipelined WGMMA."""
import hashlib,importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r2_resident_source',H.parent/'sliding_round2/kernels.py');r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
PREP=r'''
template<int QT>
DI void prep_verify(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  constexpr int L=8,H=8/QT,NR=L*H,R=4,ROWS=L*(H+2);
  const int tid=threadIdx.x,w=tid>>5,lane=tid&31;
  for(int i=tid;i<256;i+=128) {
    if(i*16>=NR*64)*(uint4*)(Q+i*16)=make_uint4(0,0,0,0);
    if(i<64)*(uint4*)(KN+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(VN+i*16)=make_uint4(0,0,0,0);
  }
  for(int i=tid;i<256;i+=128) {
    float sn,cs;__sincosf((float)p.positions[b*8+(i>>5)]*p.inv_freq[i&31],&sn,&cs);csbuf[i]=make_float2(cs,sn);
  }
  __syncthreads();asm volatile("griddepcontrol.wait;":::"memory");
  // Bounded live row vectors avoid retaining twenty rows' address and split
  // accumulators through preparation. Both BF16 and FP32 partials are exact.
#pragma unroll 1
  for(int base=0;base<ROWS;base+=4*R) {
    int tk[R],col[R];bool valid[R];float2 x[R];
#pragma unroll
    for(int j=0;j<R;j++) {
      const int row=base+w+4*j,tok=row/(H+2),kind=row%(H+2);
      tk[j]=b*8+min(tok,7);valid[j]=tok<8;
      col[j]=kind<H?(kh*8+sub*H+kind)*64+2*lane:
             kind==H?(p.HQ+kh)*64+2*lane:(p.HQ+p.HK+kh)*64+2*lane;
    }
    input_pairs<R>(p,tk,col,valid,x);
#pragma unroll
    for(int j=0;j<R;j++) {
      const int row=base+w+4*j,tok=row/(H+2),kind=row%(H+2);float2 y=x[j];
      if(kind<=H) {
        const float norm=rsqrtf(wsum(y.x*y.x+y.y*y.y)*(1.f/64.f)+1e-5f);
        const float2 cs=csbuf[min(tok,7)*32+lane];const float a=y.x*norm,z=y.y*norm;
        y=make_float2(a*cs.x-z*cs.y,a*cs.y+z*cs.x);
      }
      if(tok<8) {
        if(kind<H)*(uint16_t*)(Q+off(tok*H+kind,2*lane))=fp8pair(y.x,y.y);
        else {
          const uint16_t kv=fp8pair(y.x/(kind==H?*p.ks:*p.vs),y.y/(kind==H?*p.ks:*p.vs));
          if(kind==H)*(uint16_t*)(KN+off(tok,2*lane))=kv;
          else {const int pos=4*(tok>>1)+(tok&1);VN[off(2*lane,pos)]=(uint8_t)kv;VN[off(2*lane+1,pos)]=(uint8_t)(kv>>8);}
          if(last && sub==0)*(uint16_t*)((kind==H?p.kc:p.vc)+(p.loc[b*8+tok]*p.HK+kh)*64+2*lane)=kv;
        }
      }
    }
  }
  __syncthreads();proxy_fence();
}
'''
STEP=r'''
DI void resident_step(float (&sx)[16][4],uint32_t (&pc)[16],uint32_t (&pn)[16],float (&O)[8][4],float (&m)[2],float (&ls)[2],
 const uint8_t* Q,const uint8_t* K,const uint8_t* VT,const VP& p,float sc,int it,int begin,int seq,int end) {
  float alpha[2];protect(O);guardp(pc);guard(sx);wg_fence();
  qk128(sx,Q,K+(it+1)*8192);wg_commit();
  wg_fence();multiply_v<128>(O,pc,VT+it*8192);wg_commit();
  wg_wait<1>();guard(sx);
  prob<8,1,128>(sx,m,ls,alpha,pn,sc,begin+(it+1)*128,seq,end,p.window);
  wg_wait<0>();protect(O);guardp(pc);rescale(O,alpha);protect(O);
}
'''


def source(cs=4,pipeline=True,local=False):
    assert cs in (2,4,8,16)
    if local:assert cs>=4,'sm120 C2 uses explicit C4 math/reduction stand-in'
    cap=2048//cs;s=r2.source('last' if local else 'dsm')
    s=s.replace('constexpr int NSMAX = 128;',f'constexpr int CS={cs},CAP={cap};\nconstexpr int NSMAX = 128;')
    a=s.index('template<int QT>\nDI void prep_verify');b=s.index('template<int L,int QT,int NK>\nDI void prob(',a);s=s[:a]+PREP+s[b:]
    a=s.index('template<int L,int QT,int NK> constexpr int cluster_smem()')
    s=s[:a]+'''DI int hist_i32(const int* p){int x;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;}
'''+(STEP if pipeline else '')+s[a:]
    s=s.replace('return 4096+5*NK*64+8192+L*32*8;', 'return 4096+2*CAP*64+5120+L*32*8;')
    s=s.replace('uint8_t* Q=sm;uint8_t* K=Q+4096;uint8_t* V=K+2*KB;uint8_t* VT=V+2*KB;',
                'uint8_t* Q=sm;uint8_t* K=Q+4096;uint8_t* V=K+CAP*64;uint8_t* VT=V;')
    s=s.replace('uint8_t* KN=VT+KB;uint8_t* VN=KN+4096;float2* csbuf=(float2*)(VN+4096);',
                'uint8_t* KN=V+CAP*64;uint8_t* VN=KN+1024;float2* csbuf=(float2*)(VN+4096);')
    s=s.replace('const int seq=p.seq[b],lo=', 'const int seq=hist_i32(p.seq+b),lo=')
    a=s.index('  auto issue=[&]');b=s.index('  if constexpr(L==1)',a)
    s=s[:a]+r'''
  static_assert(CAP*128+1024>=64*68*4,"drained KV and KN storage holds all DSM partials");
  // Every history row owned by this split is resident before producer wait.
  // Page loads and asynchronous KV copies are volatile PTX, not compiler hints.
  constexpr int R=CAP/32;int pages[R];
#pragma unroll
  for(int j=0;j<R;j++) {
    const int key=min(begin+tid/4+32*j,max(seq-1,0));
    pages[j]=hist_i32(p.pt+(int64_t)b*p.pages+(key>>p.lg));
  }
#pragma unroll
  for(int j=0;j<R;j++) {
    const int row=tid/4+32*j,key=begin+row,clamped=min(key,max(seq-1,0));
    const int physical=pages[j]*p.ps+(clamped&(p.ps-1));
    const int64_t offg=((int64_t)physical*p.HK+kh)*64+16*(tid&3);
    cp16(K+off(row,16*(tid&3)),p.kc+offg,key<end);
    cp16(V+off(row,16*(tid&3)),p.vc+offg,key<end);
    if((j&3)==3)commit();
  }
  cp_wait<0>();__syncthreads();
  // Transpose each slab through 4KiB scratch, then retain VT in-place in V.
  // The scratch subsequently becomes the new-token KN/VN storage.
  for(int slab=0;slab<CAP/64;slab++) {
    trans64(V+slab*4096,KN,w,lane);__syncthreads();
    for(int z=tid;z<256;z+=128)*(uint4*)(V+slab*4096+z*16)=*(uint4*)(KN+z*16);
    __syncthreads();
  }
  proxy_fence();
'''+s[b:]
    a=s.index('  for(int it=0;it<nt;it++)',s.index('__global__ __launch_bounds__'));b=s.index('  if(split==p.ns-1)virtual_tile',a)
    if pipeline:
        body='''  if(nt) {
    float sx[16][4];uint32_t pa[16],pb[16];
    guard(sx);wg_fence();qk128(sx,Q,K);wg_commit();wg_wait<0>();guard(sx);
    prob<8,1,128>(sx,m,ls,alpha,pa,sc,begin,seq,end,p.window);rescale(O,alpha);protect(O);
    int it=0;
#pragma unroll 1
    for(;it+2<nt;it+=2) {
      resident_step(sx,pa,pb,O,m,ls,Q,K,VT,p,sc,it,begin,seq,end);
      resident_step(sx,pb,pa,O,m,ls,Q,K,VT,p,sc,it+1,begin,seq,end);
    }
    if(it+1<nt) {
      resident_step(sx,pa,pb,O,m,ls,Q,K,VT,p,sc,it,begin,seq,end);
      protect(O);guardp(pb);wg_fence();multiply_v<128>(O,pb,VT+(nt-1)*8192);wg_commit();wg_wait<0>();protect(O);guardp(pb);
    } else {
      protect(O);guardp(pa);wg_fence();multiply_v<128>(O,pa,VT+(nt-1)*8192);wg_commit();wg_wait<0>();protect(O);guardp(pa);
    }
  }
  __syncthreads();
'''
    else:
        body='''  for(int it=0;it<nt;it++) {
    float s[16][4];uint32_t pk[16];guard(s);wg_fence();qk128(s,Q,K+it*8192);wg_commit();wg_wait<0>();guard(s);
    prob<8,1,128>(s,m,ls,alpha,pk,sc,begin+it*128,seq,end,p.window);
    rescale(O,alpha);protect(O);guardp(pk);wg_fence();multiply_v<128>(O,pk,VT+it*8192);wg_commit();wg_wait<0>();protect(O);guardp(pk);
  }
  __syncthreads();
'''
    s=s[:a]+body+s[b:]
    if not local:
        # Constant gang size collapses the epilogue's sixteen-slot live arrays.
        a=s.index('  // All asynchronous use has drained;');b=s.index('\n}\n\ntemplate<int L,int QT,int NK> void attrs()',a)
        ep=s[a:b].replace('p.cs','CS').replace('float2 ml[16],ov[16]','float2 ml[CS],ov[CS]').replace('c<16','c<CS')
        s=s[:a]+ep+s[b:]
        s=s.replace('int qt, ns, cs, qwidth;', 'int* count; int qt, ns, cs, qwidth;')
    a=s.index('__global__ __launch_bounds__');b=s.index('template<int L,int QT,int NK> void attrs()',a)
    s=s[:a]+s[a:b].replace('p.ns','CS')+s[b:]
    a=s.index('template<int L,int QT,int NK> void run(VP p,cudaStream_t stream)')
    s=s[:a]+(H/'early_edge.cu.inc').read_text()+'\n'+s[a:]
    s=s.replace('void run(VP p,cudaStream_t stream) {','void run(VP p,cudaStream_t stream,bool early) {')
    a=s.index('  auto e=cudaLaunchKernelEx',s.index('void run(VP p,'))
    edge='''  void* args[]={&p};cudaKernelNodeParams node={};node.func=(void*)sliding_main<L,QT,NK>;node.gridDim=cfg.gridDim;node.blockDim=cfg.blockDim;node.sharedMemBytes=cfg.dynamicSmemBytes;node.kernelParams=args;
  if(vr_capture_consumer(stream,node,CLUSTER,early))return;
'''.replace('CLUSTER','1' if local else 'CS')
    s=s[:a]+edge+s[a:]
    # One ABI for native DSM and explicit local global-last-CTA reduction.
    a=s.index('std::vector<int64_t> sliding_resources(')
    s=s[:a]+r'''
std::vector<int64_t> sliding_resources(int64_t L,int64_t qt,int64_t nk,int64_t cs) {
  if(L!=8 || qt!=1 || nk!=128 || cs!=CS)throw std::runtime_error("resident verify shape mismatch");
  return resource<8,1,128>(cs);
}
void sliding_forward(int64_t qkv,int64_t positions,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t inv,int64_t out,
 int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,int64_t splits,int64_t split_stride,int64_t qt,int64_t nk,int64_t cs,int64_t width,double sc,int64_t stream,int64_t part,int64_t counter,int64_t ns,int64_t early) {
  if(L!=8 || qt!=1 || nk!=128 || cs!=CS || ns!=CS || window>2048)throw std::runtime_error("resident verify shape mismatch");
  VP p={};p.part=(float*)part;p.count=(int*)counter;p.qkv=(const void*)qkv;p.positions=(const int64_t*)positions;p.loc=(const int64_t*)loc;p.pt=(const int*)pt;p.seq=(const int*)seq;
  p.kc=(uint8_t*)kc;p.vc=(uint8_t*)vc;p.ks=(const float*)ks;p.vs=(const float*)vs;p.inv_freq=(const float*)inv;p.out=(__nv_bfloat16*)out;
  p.B=B;p.L=L;p.HQ=HQ;p.HK=HK;p.stride=stride;p.pages=pages;p.ps=ps;p.window=window;p.proj_splits=splits;p.split_stride=split_stride;p.qt=1;p.ns=CS;p.cs=CS;
  p.gate=width>(HQ+2*HK)*64;p.sc=sc;while((1<<p.lg)<ps)p.lg++;
  run<8,1,128>(p,(cudaStream_t)stream,early);
  auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
'''
    if local:s=s.replace('return resource<8,1,128>(cs);','return resource<8,1,128>(1);')
    assert 'issue(' not in s
    return s

if __name__=='__main__':
    (H/'out').mkdir(exist_ok=True)
    for cs,pipe in ((2,True),(4,True),(8,True),(16,True),(4,False),(8,False)):
        s=source(cs,pipe);p=H/'out'/f'c{cs}_{"p" if pipe else "s"}.cu';p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
