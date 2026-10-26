"""Independent S15 heavy PACK5, block-readiness and measured CHUNK64 arms."""
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

H=Path(__file__).resolve().parent;B=H.parent;ROOT=B.parents[4]
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best7_s15.py'
SHA='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
P3A=ROOT/'problems/muse_decode_layer/dev/nt1_timeline/review'
sys.path.insert(0,str(B))
from light_build import value

def prepare_parent():
 spec=importlib.util.spec_from_file_location('p3b_s15_parent',B/'round6/build.py')
 parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent);parent.main()

def function(src,name):
 return ast.get_source_segment(src,next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name))

def once(s,a,b):
 assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)

def chunk(cu):
 # Same native tile and kernel as the gated R6 receipt; only the host dyn helper's namespace changes.
 s=(B/'round6/repair.cu').read_text();s=s[s.index('namespace chunk {'):].replace('oc::dyn<','chunk::dyn<')
 return '#include <cuda.h>\n'+cu+'\n'+s

def ready(cu):
 cu=once(cu,'template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>','template<int NT,bool FIRST,int VEC,int FM,int LO,int HI,bool READY=false>')
 cu=once(cu,'float*SS,int M,unsigned scale_base){','float*SS,int M,unsigned scale_base,int*READY_COUNT,int RB){')
 cu=once(cu,'  float acc[NT/2]={};', '''  if constexpr(READY&&!FIRST){
   unsigned done;do{asm volatile("ld.global.acquire.gpu.u32 %0,[%1];":"=r"(done):"l"(READY_COUNT+e*RB+row/32):"memory");if(done<128)__nanosleep(64);}while(done<128);
   __syncthreads();
  }
  float acc[NT/2]={};''')
 # Publish after every writer has fenced its ACT stores, then join the CTA's visibility frontier.
 cu=once(cu,'  __syncthreads();\n }\n}\n\n// Routing only:', '''  if constexpr(READY&&FIRST){__threadfence();__syncthreads();if(t==0)atomicAdd(READY_COUNT+e*RB+row/32,1);}
  __syncthreads();
 }
}

// Routing only:''')
 cu=once(cu,'(BR_ARGS(v),(int)M,0x70007000u);break;', '(BR_ARGS(v),(int)M,0x70007000u,nullptr,0);break;')
 cu+='''
template<bool F>std::vector<int64_t> ready_resource(){cudaFuncAttributes a{};int n;
 auto kernel=mwbr::gemm<32,F,F?7:6,0,0,0,true>;
 BR_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,mwbr::smem_bytes<32>()));
 BR_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
 BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,kernel,128,mwbr::smem_bytes<32>()));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)mwbr::smem_bytes<32>(),n};}
std::vector<int64_t> ready_info(int64_t f){return f?ready_resource<true>():ready_resource<false>();}
void ready_run(std::vector<int64_t>v,int64_t first,int64_t grid,int64_t M,int64_t stream,int64_t counter,int64_t rb){
 if(first)mwbr::gemm<32,true,7,0,0,0,true><<<grid,128,mwbr::smem_bytes<32>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,0x70007000u,(int*)counter,(int)rb);
 else mwbr::gemm<32,false,6,0,0,0,true><<<grid,128,mwbr::smem_bytes<32>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,0x70007000u,(int*)counter,(int)rb);
 BR_CUDA_CHECK(cudaGetLastError());}
'''
 return cu

def pack(cu):
 helper=(P3A/'pack5_cuda.py.inc').read_text()
 # The P3a packet format is byte-exact; here codes retain S15's wide/interleaved order.
 cu=once(cu,'namespace mwbr {',helper+'\nnamespace mwbr {')
 cu=once(cu,'constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;',
 '''constexpr bool PACK5=FM==0;
 constexpr int QB=64*KT/2,SB=PACK5?576:64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;''')
 cu=once(cu,'const int*qg=Q;const unsigned char*sg=SC;', 'const int*qg=Q;const unsigned char*sg=SC;unsigned rawmask=0;int sfstep=0;')
 cu=once(cu,'if(wq){unsigned gp=t>>4,ln=t&15;qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;}', '''if(wq){unsigned gp=t>>4,ln=t&15;
    if constexpr(PACK5){int bytes;const auto*tile=packet_tile((const uint8_t*)Q,(unsigned)e*(N/16)+mt*4+gp,bytes);qg=(const int*)tile+ln*4;}
    else qg=Q+((unsigned)e*(N/16)+mt*4+gp)*128*64+ln*4;
   }''')
 cu=once(cu,'if(ws){unsigned i=t-64;if constexpr(WIDE)sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);', '''if(ws){unsigned i=t-64;if constexpr(PACK5){int bytes;sg=packet_tile(SC,(unsigned)e*(N/16)+mt*4+i/8,bytes)+32768;rawmask=*(const unsigned*)(sg+16);}
    else if constexpr(WIDE)sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);''')
 cu=once(cu,'if constexpr(WIDE)sd=smaddr(ss+i*16);','if constexpr(PACK5)sd=smaddr(ss+(i/8)*144+(i%8)*16);else if constexpr(WIDE)sd=smaddr(ss+i*16);')
 cu=once(cu,'if(ws){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}', '''if(ws){if constexpr(PACK5){unsigned i=t-64,ln=i&7,p=sfstep>>1;bool raw=(rawmask>>p)&1;
      unsigned start=p*160+__popc(rawmask&((1u<<p)-1u))*96+(sfstep&1)*(raw?128:80);
      if(raw||ln<5)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg+32+start+ln*16):"memory");
      if(ln==0)*(unsigned*)(ss+slot*SB+(i/8)*144+128)=((unsigned)raw<<8)|sg[p];++sfstep;
     }else{asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+slot*SB),"l"(sg):"memory");sg+=(KT/32)*32;}}
''')
 cu=once(cu,'uint4 sv;if constexpr(WIDE)sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);', '''uint4 sv;if constexpr(PACK5)sv=packet_sv(ss+st*SB+w*144,l>>2);
     else if constexpr(WIDE)sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);''')
 decode='''
__device__ __forceinline__ unsigned spread5(unsigned x,unsigned base){
 unsigned y=(x&31u)|((x&0x3e0u)<<3)|((x&0x7c00u)<<6)|((x&0xf8000u)<<9);
 return y+base*0x01010101u;
}
__device__ __forceinline__ uint4 packet_sv(const unsigned char*p,int q){
 unsigned meta=*(const unsigned*)(p+128);uint4 v;
 if(meta&256u)v=*(const uint4*)(p+q*16);
 else{int bit=q*80;auto*a=(const unsigned*)(p+(bit>>5)*4);unsigned x=a[0],y=a[1],z=a[2],base=meta&255u;
  if(bit&31){v.x=spread5(__funnelshift_r(x,y,16),base);v.y=spread5(y>>4,base);v.z=spread5(__funnelshift_r(y,z,24),base);v.w=spread5(z>>12,base);}
  else{v.x=spread5(x,base);v.y=spread5(__funnelshift_r(x,y,20),base);v.z=spread5(__funnelshift_r(y,z,8),base);v.w=spread5(__funnelshift_r(y,z,28),base);}}
 // Recreate exactly the timed decoder's original PERMW word rotation.
 v.x=__funnelshift_l(v.x,v.x,4);v.y=__funnelshift_l(v.y,v.y,4);v.z=__funnelshift_l(v.z,v.z,4);v.w=__funnelshift_l(v.w,v.w,4);return v;
}
'''
 cu=once(cu,'namespace mwbr {','namespace mwbr {\n'+decode)
 # PACK5 has a slightly larger stage metadata slot; light still uses the original size.
 cu=cu.replace('mwbr::smem_bytes<NT>()','(FM==0?52224:mwbr::smem_bytes<NT>())')
 return cu

def source(arm):
 base=BASE.read_text();assert hashlib.sha256(base.encode()).hexdigest()==SHA
 cu=value(base,'NF_CUDA');cpp=value(base,'NF_CPP');fs=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run']
 plan=function(base,'nf_plan').replace('def nf_plan(','def p3b_plan(')
 extra='';bind='p3b_module()\n'
 if arm in ('CHUNK3','CHUNK4'):
  cu=chunk(cu);fs+=['ch_run','ch_info'];cpp+='void ch_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> ch_info(int64_t,int64_t);'
  depth=int(arm[-1]);plan=plan.replace('mod.br_run(p13,26,grid,m,side.cuda_stream);mod.br_run(p2,21,grid,m,side.cuda_stream)',f'mod.ch_run(p13,1,{depth},sms*4,m,side.cuda_stream);mod.ch_run(p2,0,{depth},sms*4,m,side.cuda_stream)')
  bind+=f'p3b_module().ch_info({depth},0);p3b_module().ch_info({depth},1)\n'
 elif arm=='PACK5':
  cu=pack(cu);extra=(P3A/'pack5_python.py.inc').read_text().replace('pack5_','hp5_')
  extra+='''
def p3b_packets(base):
 result={}
 for k in ('w13','w2'):
  q=base[k].contiguous().view(torch.uint8).reshape(-1,32768)
  sf=base[k+'_sf'].contiguous().reshape(-1).view(torch.int32).to(torch.int64)&0xffffffff
  sf=(((sf>>4)|(sf<<28))&0xffffffff).to(torch.int32).view(torch.uint8).reshape(-1,4096)
  result[k]=hp5_bank(torch.cat((q,sf),dim=1).contiguous())
 return result
'''
  plan=plan.replace('def p3b_plan(data,base,native,streams):','def p3b_plan(data,base,native,streams,packets):')
  plan=once(plan,'    p13,p2,pn=',"    h13[0]=packets['w13'];h13[1]=packets['w13'];h2[0]=packets['w2'];h2[1]=packets['w2']\n    p13,p2,pn=")
  plan=once(plan,'    l13=list(h13);l2=list(h2);',"    l13=list(h13);l2=list(h2);l13[:2]=[base['w13'],base['w13_sf']];l2[:2]=[base['w2'],base['w2_sf']];")
  bind+="state['p3b_packets']=p3b_packets(state['s5_hh_base'])\n"
 elif arm in ('READY','READY_SERIAL'):
  if arm=='READY':
   cu=ready(cu);fs+=['ready_run','ready_info'];cpp+='void ready_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> ready_info(int64_t);'
   plan=once(plan,'side,fork,join,light,ljoin=streams;', 'side,fork,join,light,ljoin,w2side,w2join=streams;')
   plan=once(plan,'    stages=2', '    rb=(m+31)//32;ready_count=torch.empty((32,rb),device=dev,dtype=torch.int32)\n    stages=2')
   plan=once(plan,'        _=owners;mod.br_route', '        _=owners;ready_count.zero_();mod.br_route')
   plan=once(plan,'mod.br_run(p13,26,grid,m,side.cuda_stream);mod.br_run(p2,21,grid,m,side.cuda_stream);join.record()', 'mod.ready_run(p13,1,grid,m,side.cuda_stream,ready_count.data_ptr(),rb);join.record()')
   plan=once(plan,'        with torch.cuda.stream(light):', '''        w2side.wait_event(fork)
        with torch.cuda.stream(w2side):
            mod.ready_run(p2,0,(sms if torch.cuda.get_device_capability()==(9,0) else max(1,sms//2)),m,w2side.cuda_stream,ready_count.data_ptr(),rb);w2join.record()
        with torch.cuda.stream(light):''')
   plan=once(plan,'parent.wait_event(join);parent.wait_event(ljoin)', 'parent.wait_event(join);parent.wait_event(ljoin);parent.wait_event(w2join)')
   bind+="state['nf_streams']=(*state['nf_streams'],torch.cuda.Stream(),torch.cuda.Event())\np3b_module().ready_info(0);p3b_module().ready_info(1)\n"
  else:plan=once(plan,'mod.br_run(p2,21,grid,m,side.cuda_stream)', 'mod.br_run(p2,21,(sms if torch.cuda.get_device_capability()==(9,0) else max(1,sms//2)),m,side.cuda_stream)')
 else:raise ValueError(arm)
 plan=once(plan,'mod=nf_module()','mod=p3b_module()')
 cell='P3B_CUDA='+repr(cu)+'\nP3B_CPP='+repr(cpp)+'\n_P3B_MODULE=[]\n'
 cell+='''def p3b_module():
 if not _P3B_MODULE:
  target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
  mod=load_inline(name='p3b_s15_'+hashlib.sha256(P3B_CUDA.encode()).hexdigest()[:12],cpp_sources=P3B_CPP,cuda_sources=P3B_CUDA,functions=P3B_FUNCTIONS,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target])
  mod.br_init();NF_OPTS['pmz']=mod.pmz_init();_P3B_MODULE.append(mod)
 return _P3B_MODULE[0]
'''
 cell+='P3B_FUNCTIONS='+repr(fs)+'\n'+extra+'\n'+plan+'\n'
 cell+='def p3b_bind(state):\n'+''.join(' '+line+'\n' for line in bind.splitlines())+' return state\n'
 args="data,state['s5_hh_base'],state['moe_bank'],state['nf_streams']"+( ",state['p3b_packets']" if arm=='PACK5' else '')
 cell+=f'''_p3b_original_nf_call=nf_call
def nf_call(data,state):
 for fn in p3b_plan({args})['steps']:out=fn()
 return out
_p3b_original_s4_prepare_banks=_s4_prepare_banks
def _s4_prepare_banks(state,weights):
 _p3b_original_s4_prepare_banks(state,weights)
 return p3b_bind(state)
'''
 ast.parse(cell)
 return base,cell,cu,fs

def main():
 prepare_parent()
 manifest={}
 out=ROOT/'runs/screen/p3b_s15_review';out.mkdir(parents=True,exist_ok=True)
 for arm in ('CHUNK3','CHUNK4','PACK5','READY_SERIAL','READY'):
  base,cell,cu,fs=source(arm);candidate=base+'\n'+cell;p=out/(arm.lower()+'.py');p.write_text(candidate)
  (H/(arm.lower()+'_cell.py')).write_text(cell);(H/(arm.lower()+'.cu')).write_text(cu)
  patch=''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+str(BASE.relative_to(ROOT)),tofile='b/'+str(BASE.relative_to(ROOT))))
  (H/(arm.lower()+'.patch')).write_text(patch)
  manifest[arm]=dict(base_sha256=SHA,candidate=str(p.relative_to(ROOT)),candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),cell=str((H/(arm.lower()+'_cell.py')).relative_to(ROOT)),cell_sha256=hashlib.sha256(cell.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),functions=fs)
 (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)

if __name__=='__main__':main()
