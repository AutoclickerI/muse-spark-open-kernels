"""Single-launch sliding: token-aware last-CTA, cooperative verify prep, early-history and DSM arms."""
import hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cluster_sliding'))
import build_last,build_gated,build
VERIFY=r'''
template<int QT>
DI void prep_verify(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  constexpr int L=8,H=8/QT,NR=L*H,R=(L*(H+2)+3)/4;
  const int tid=threadIdx.x,w=tid>>5,lane=tid&31;
  // Input metadata and zero padding are independent of the QKV producer.
  for(int i=tid;i<256;i+=128) {
    if(i*16>=NR*64)*(uint4*)(Q+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(KN+i*16)=make_uint4(0,0,0,0);*(uint4*)(VN+i*16)=make_uint4(0,0,0,0);
  }
  for(int i=tid;i<8*32;i+=128) {
    float sn,cs;__sincosf((float)p.positions[b*8+(i>>5)]*p.inv_freq[i&31],&sn,&cs);
    csbuf[i]=make_float2(cs,sn);
  }
  __syncthreads();asm volatile("griddepcontrol.wait;":::"memory");
  int tk[R],col[R];bool valid[R];float2 x[R];
#pragma unroll
  for(int j=0;j<R;j++) {
    const int row=w+4*j,tok=row/(H+2),kind=row%(H+2);
    tk[j]=b*8+min(tok,7);valid[j]=tok<8;
    col[j]=kind<H?(kh*8+sub*H+kind)*64+2*lane:
           kind==H?(p.HQ+kh)*64+2*lane:(p.HQ+p.HK+kh)*64+2*lane;
  }
  input_pairs<R>(p,tk,col,valid,x);
#pragma unroll
  for(int j=0;j<R;j++) {
    const int row=w+4*j,tok=row/(H+2),kind=row%(H+2);
    float2 y=x[j];
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
  __syncthreads();proxy_fence();
}
'''

def source(kind='last'):
    assert kind in ('last','early','dsm')
    s=build_gated.source() if kind=='dsm' else build_last.source()
    a=s.index('template<int L,int QT,int NK>\nDI void prob(')
    s=s[:a]+VERIFY+s[a:]
    s=s.replace('else prep<L,QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);',
                'else prep_verify<QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);')
    if kind!='dsm':
        s=s.replace('if(L!=1 || ns<1 || ns>32)','if((L!=1 && L!=8) || ns<1 || ns>32)')
        s=s.replace('  else throw std::runtime_error("decode only");','  else {if(qt==4){RES(8,4);}else{RES(8,1);}}')
        at=s.index('\n#undef GO');s=s[:at]+'\n  else {if(qt==4){GO(8,4);}else{GO(8,1);}}'+s[at:]
        s=s.replace('const int head=kh*8+sub*H+r;float2 gate=make_float2(0.f,0.f);',
                    'const int token=r/H,head=kh*8+sub*H+r%H;float2 gate=make_float2(0.f,0.f);')
        s=s.replace('if(p.gate)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);', '''if(p.gate) {
      if constexpr(L==1)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);
      else {int tk[1]={b*L+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 gv[1];input_pairs<1>(p,tk,col,valid,gv);gate=gv[0];}
    }''')
        s=s.replace('((int64_t)b*p.HQ+head)*64','((int64_t)(b*L+token)*p.HQ+head)*64')
    # Explicit query subgroups shorten the last owner's verify row-merge tail.
    for L in (1,8):
        for macro in ('RES','GO'):
            old=f'if(qt==4){{{macro}({L},4);}}else{{{macro}({L},1);}}'
            new=f'if(qt==8){{{macro}({L},8);}}else if(qt==4){{{macro}({L},4);}}else if(qt==2){{{macro}({L},2);}}else{{{macro}({L},1);}}'
            assert old in s,old;s=s.replace(old,new)
    if kind=='early':
        # All 2047 history keys are fetched by 16 splits, one tile each.
        s=s.replace('return 4096+5*NK*64+8192+L*32*8;', 'return 4096+3*NK*64+L*32*8;')
        s=s.replace('uint8_t* Q=sm;uint8_t* K=Q+4096;uint8_t* V=K+2*KB;uint8_t* VT=V+2*KB;',
                    'uint8_t* Q=sm;uint8_t* K=Q+4096;uint8_t* V=K+KB;uint8_t* VT=V+KB;')
        s=s.replace('uint8_t* KN=VT+KB;uint8_t* VN=KN+4096;float2* csbuf=(float2*)(VN+4096);',
                    'uint8_t* KN=V;uint8_t* VN=V+4096;float2* csbuf=(float2*)(VT+KB);')
        s=s.replace('const int seq=p.seq[b],lo=', 'const int seq=hist_i32(p.seq+b),lo=')
        s=s.replace('pages[j]=p.pt[(int64_t)b*p.pages+(clamped>>p.lg)];','pages[j]=hist_i32(p.pt+(int64_t)b*p.pages+(clamped>>p.lg));')
        s=s.replace('K+slot*KB+off(', 'K+off(').replace('V+slot*KB+off(', 'V+off(')
        s=s.replace('  issue(0,0);issue(1,1);', '''  issue(0,0);cp_wait<0>();__syncthreads();
#pragma unroll
  for(int j=0;j<NK/64;j++)trans64(V+j*4096,VT+j*4096,w,lane);
  __syncthreads();proxy_fence();''')
        a=s.index('    cp_wait<1>();__syncthreads();',s.index('__global__ __launch_bounds__'));b=s.index('    float s[NK/8][4];',a)
        s=s[:a]+s[b:];s=s.replace('K+(it%2)*KB','K').replace('    __syncthreads();issue(it+2,it%2);','    __syncthreads();')
        s=s.replace('if((L!=1 && L!=8) || ns<1 || ns>32)', 'if((L!=1 && L!=8) || nk!=128 || ns!=16 || window>2048)')
        s=s.replace('if(nk==64)return resource<L,Q,64>(cs);else return resource<L,Q,128>(cs)', 'return resource<L,Q,128>(cs)')
        s=s.replace('if(nk==64)run<L,Q,64>(p,(cudaStream_t)stream);else run<L,Q,128>(p,(cudaStream_t)stream)', 'run<L,Q,128>(p,(cudaStream_t)stream)')
        a=s.index('template<int QT>\nDI void prep_verify')
        s=s[:a]+'''DI int hist_i32(const int* p) {int x;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;}
'''+s[a:]
    # Instantiate only N128 for the actual batch; no unused alternative bodies.
    if kind!='early':
        s=s.replace('if(nk==64)return resource<L,Q,64>(cs);else return resource<L,Q,128>(cs)', 'return resource<L,Q,128>(cs)')
        s=s.replace('if(nk==64)run<L,Q,64>(p,(cudaStream_t)stream);else run<L,Q,128>(p,(cudaStream_t)stream)', 'run<L,Q,128>(p,(cudaStream_t)stream)')
    return s

if __name__=='__main__':
    (HERE/'out').mkdir(exist_ok=True)
    for k in ('last','early','dsm'):
        s=source(k);(HERE/'out'/(k+'.cu')).write_text(s);print(k,hashlib.sha256(s.encode()).hexdigest())
