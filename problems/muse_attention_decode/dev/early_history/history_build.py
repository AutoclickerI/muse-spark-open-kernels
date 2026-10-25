"""One-tile sliding history, retained before PDL wait; compact shared-memory lifetime."""
import hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cluster_sliding'))
import build_last

def source(early=True,clock=False):
    s=build_last.source()
    # One history tile/split: the driver enforces 16 splits for a <=2048 window.
    # Raw V dies before KN/VN preparation and shares their 8KiB allocation.
    s=s.replace('return 4096+5*NK*64+8192+L*32*8;', 'return 4096+3*NK*64+L*32*8;')
    s=s.replace('uint8_t* Q=sm;uint8_t* K=Q+4096;uint8_t* V=K+2*KB;uint8_t* VT=V+2*KB;',
                'uint8_t* Q=sm;uint8_t* K=Q+4096;uint8_t* V=K+KB;uint8_t* VT=V+KB;')
    s=s.replace('uint8_t* KN=VT+KB;uint8_t* VN=KN+4096;float2* csbuf=(float2*)(VN+4096);',
                'uint8_t* KN=V;uint8_t* VN=V+4096;float2* csbuf=(float2*)(VT+KB);')
    # Pin page-table and metadata reads as volatile PTX, never dependent QKV.
    s=s.replace('const int seq=p.seq[b],lo=', 'const int seq=hist_i32(p.seq+b),lo=')
    s=s.replace('pages[j]=p.pt[(int64_t)b*p.pages+(clamped>>p.lg)];',
                'pages[j]=hist_i32(p.pt+(int64_t)b*p.pages+(clamped>>p.lg));')
    s=s.replace('  auto issue=[&](int tile,int slot) {', '  auto issue=[&](int tile,int slot) {')
    s=s.replace('K+slot*KB+off(', 'K+off(').replace('V+slot*KB+off(', 'V+off(')
    a=s.index('template<int QT,bool GATE>');b=s.index('template<int L,int QT,int NK>\nDI void prob(',a)
    prep=s[a:b];assert prep.count('if(w==0)csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);')==1
    prep=prep.replace('if(w==0)csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);','')
    s=s[:a]+prep+s[b:]
    s=s.replace('  issue(0,0);issue(1,1);', '''  issue(0,0);
  if(w==0)csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);
  cp_wait<0>();__syncthreads();
#pragma unroll
  for(int j=0;j<NK/64;j++)trans64(V+j*4096,VT+j*4096,w,lane);
  __syncthreads();proxy_fence();''')
    # The "late" control retains compact layout but completes/transposes after
    # producer synchronization, before preparation overwrites raw V.
    if not early:
        s=s.replace('  issue(0,0);', '  asm volatile("griddepcontrol.wait;":::"memory");\n  issue(0,0);')
    a=s.index('    cp_wait<1>();__syncthreads();',s.index('__global__ __launch_bounds__'))
    b=s.index('    float s[NK/8][4];',a)
    s=s[:a]+s[b:]
    s=s.replace('K+(it%2)*KB','K')
    s=s.replace('    __syncthreads();issue(it+2,it%2);','    __syncthreads();')
    # Refuse oversized windows instead of silently dropping history.
    s=s.replace('if(L!=1 || ns<1 || ns>32)', 'if(L!=1 || window>ns*nk || nk!=128 || ns<16 || ns>32)')
    helper='''DI int hist_i32(const int* p) {
  int x;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;
}
'''
    at=s.index('template<int QT,bool GATE>');s=s[:at]+helper+s[at:]
    # Only N128 is admitted by the one-tile layout: dead N64 templates would
    # alias raw V and VT because KN/VN need 8192B irrespective of NK.
    s=s.replace('if(nk==64)return resource<L,Q,64>(cs);else return resource<L,Q,128>(cs)', 'return resource<L,Q,128>(cs)')
    s=s.replace('if(nk==64)run<L,Q,64>(p,(cudaStream_t)stream);else run<L,Q,128>(p,(cudaStream_t)stream)', 'run<L,Q,128>(p,(cudaStream_t)stream)')
    s=s.replace('if(qt==4){RES(1,4);}else{RES(1,1);}', 'if(qt==4){RES(1,4);}else if(qt==2){RES(1,2);}else{RES(1,1);}')
    s=s.replace('if(qt==4){GO(1,4);}else{GO(1,1);}', 'if(qt==4){GO(1,4);}else if(qt==2){GO(1,2);}else{GO(1,1);}')
    if clock:
        s=s.replace('float* part; __nv_bfloat16* out;', 'unsigned long long* hist_clock; float* part; __nv_bfloat16* out;')
        a=s.index('DI uint32_t su(')
        s=s[:a]+'''DI void hist_stamp(const VP& p,int phase) {
  if(threadIdx.x==0){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");
    p.hist_clock[(((blockIdx.z*gridDim.y)+blockIdx.y)*gridDim.x+blockIdx.x)*8+phase]=t;}
}
'''+s[a:]
        s=s.replace('int64_t counter,int64_t ns) {','int64_t counter,int64_t ns,int64_t clockout) {')
        s=s.replace('VP p={};p.part=', 'VP p={};p.hist_clock=(unsigned long long*)clockout;p.part=')
        s=s.replace('  asm volatile("griddepcontrol.launch_dependents;', '  hist_stamp(p,0);\n  asm volatile("griddepcontrol.launch_dependents;')
        s=s.replace('  issue(0,0);','  issue(0,0);hist_stamp(p,1);')
        s=s.replace('  cp_wait<0>();__syncthreads();\n#pragma unroll','  cp_wait<0>();__syncthreads();hist_stamp(p,2);\n#pragma unroll')
        s=s.replace('  __syncthreads();proxy_fence();\n  if constexpr(L==1)', '  __syncthreads();proxy_fence();hist_stamp(p,3);\n  if constexpr(L==1)')
        s=s.replace('  asm volatile("griddepcontrol.wait;":::"memory");', '  asm volatile("griddepcontrol.wait;":::"memory");hist_stamp(p,4);')
        s=s.replace('  float O[8][4]={}', '  hist_stamp(p,5);\n  float O[8][4]={}')
        s=s.replace('  // Publish only live query rows.', '  hist_stamp(p,6);\n  // Publish only live query rows.')
        s=s.replace('  __syncthreads();if(tid==0)atomicExch(p.count+group*32,0);', '  hist_stamp(p,7);\n  __syncthreads();if(tid==0)atomicExch(p.count+group*32,0);')
        assert 'hist_stamp(p,7)' in s
    return s

if __name__=='__main__':
    (HERE/'out').mkdir(exist_ok=True)
    for name,early,clock in [('early',True,False),('late',False,False),('clock',True,True)]:
        s=source(early,clock);(HERE/'out'/f'{name}.cu').write_text(s);print(name,hashlib.sha256(s.encode()).hexdigest())
