"""S14 NT1 epilogue: vector K reduction, independent TMA warp, and warp-owned norm accumulation."""
import ast,hashlib
from pathlib import Path
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py'
VEC=r'''
      if(warp==0){
        float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
#pragma unroll
        for(int w=0;w<NCW;++w){float4 v=*reinterpret_cast<const float4*>(rb+(w*32+lane)*4);
          a0+=v.x;a1+=v.y;a2+=v.z;a3+=v.w;}
        int i=lane>>2,n=(lane&3)*2,idx=nb*8+n;
        if(!pass){float scale=gs13[e]*256.f;float g0=a0*scale,g1=a1*scale,u0=a2*scale,u1=a3*scale;
          if(idx<ne){if constexpr(Z)if(m==0&&i==0)ss[off+idx]=0.f;
            inter[(size_t)(off+idx)*HD+m*8+i]=__float2bfloat16(g0/(1.f+__expf(-g0))*u0);}
          if(idx+1<ne){if constexpr(Z)if(m==0&&i==0)ss[off+idx+1]=0.f;
            inter[(size_t)(off+idx+1)*HD+m*8+i]=__float2bfloat16(g1/(1.f+__expf(-g1))*u1);}
        }else{float scale=gs2[e]*256.f,p0=1.f+__bfloat162float(pnw[m*16+i]),p1=1.f+__bfloat162float(pnw[m*16+i+8]);
          float v0=a0*scale,v1=a1*scale,v2=a2*scale,v3=a3*scale;
          float z0=v0*p0,z1=v1*p0,z2=v2*p1,z3=v3*p1;
          if(idx<ne){y[(size_t)(off+idx)*HD+m*16+i]=ZOUT?z0:v0;y[(size_t)(off+idx)*HD+m*16+i+8]=ZOUT?z2:v2;}
          if(idx+1<ne){y[(size_t)(off+idx+1)*HD+m*16+i]=ZOUT?z1:v1;y[(size_t)(off+idx+1)*HD+m*16+i+8]=ZOUT?z3:v3;}
          float sq0=idx<ne?z0*z0+z2*z2:0.f,sq1=idx+1<ne?z1*z1+z3*z3:0.f;
          if constexpr(VNR){nv0+=sq0;nv1+=sq1;}else{
#pragma unroll
            for(int d=4;d<=16;d*=2){sq0+=__shfl_xor_sync(~0u,sq0,d);sq1+=__shfl_xor_sync(~0u,sq1,d);}
            if(lane<4){if(idx<ne)atomicAdd(ss+off+idx,sq0);if(idx+1<ne)atomicAdd(ss+off+idx+1,sq1);}
          }
        }
      }
'''

def source(mode):
 s=BASE.read_text();assert hashlib.sha256(s.encode()).hexdigest().startswith('f1c77ce6')
 a=s.index('SMALLM_VAR_CUDA = ');b=s.index('def _s4_prepare_banks(',a);cell=s[a:b]
 cuda=next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SMALLM_VAR_CUDA');old=cuda
 cuda=cuda.replace('  constexpr bool PDL =',f'  constexpr bool VEC={mode}>0&&NT==1&&!NREG,VNR={mode}==3&&NT==1&&!NREG;\n  constexpr bool PDL =')
 cuda=cuda.replace('  auto unit_body =',r'''  float nv0=0.f,nv1=0.f;
  auto flush_vec_norm=[&](){
    if constexpr(VNR){
      uint64_t t=0;if constexpr(TRACE)if(tid==0)t=tg();
      if(warp==0){float s0=nv0,s1=nv1;
#pragma unroll
        for(int d=4;d<=16;d*=2){s0+=__shfl_xor_sync(~0u,s0,d);s1+=__shfl_xor_sync(~0u,s1,d);}
        if(lane<4){int n=cur_nb*8+(lane&3)*2,ne=rt->n[cur_e],off=rt->off[cur_e];
          if(n<ne)atomicAdd(ss+off+n,s0);if(n+1<ne)atomicAdd(ss+off+n+1,s1);}
      }
      nv0=nv1=0.f;if constexpr(TRACE)if(tid==0)probe[60]+=tg()-t;
    }
  };
  auto unit_body =''')
 cuda=cuda.replace('      cur_pass = pass; cur_e = e;', '      if constexpr(VNR)if(cur_pass==1)flush_vec_norm();\n      cur_pass = pass; cur_e = e;')
 cuda=cuda.replace('  if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }', '  if constexpr (NREG) { if (cur_pass == 1) flush_norm(); }\n  if constexpr(VNR)if(cur_pass==1)flush_vec_norm();')
 cuda=cuda.replace('    if (pass == 1 && tid < 16 * NB) fw[0] =', '    if (!VEC && pass == 1 && tid < 16 * NB) fw[0] =')
 a=cuda.index('    float* rp = red +');b=cuda.index('    if constexpr (TRACE) {\n      if (tid == 0 && pass == 0 && firstp)',a)
 ep=cuda[a:b]
 ep=ep.replace('    float* rp =', '    uint64_t et=0;if constexpr(TRACE)if(tid==0)et=tg();\n    float* rp =')
 ep=ep.replace('    bar_consumers();\n    if (tid == 0) {', '    bar_consumers();\n    if constexpr(TRACE)if(tid==0)probe[52+pass]+=tg()-et;\n    if (tid == '+('(VEC?224:0)' if mode>=2 else '0')+') {\n      uint64_t pt=0;if constexpr(TRACE)pt=tg();')
 ep=ep.replace('    const float* rb =', '      if constexpr(TRACE)probe[54+pass]+=tg()-pt;\n    }\n    if constexpr(TRACE)if(tid==0)et=tg();\n    const float* rb =',1)
 # Replace the original prefetch block closing brace immediately before our inserted record.
 ep=ep.replace('    }\n      if constexpr(TRACE)probe[54+pass]', '      if constexpr(TRACE)probe[54+pass]',1)
 p0=ep.index('    if (pass == 0) {');oldbody=ep[p0:]
 ep=ep[:p0]+'    if constexpr(VEC){\n'+VEC+'    }else{\n'+oldbody+'    }\n'
 ep+='    if constexpr(TRACE)if(tid==0){probe[56+pass]+=tg()-et;probe[58+pass]++;}\n'
 cuda=cuda[:a]+ep+cuda[b:]
 cuda=cuda.replace('#define VAR_FLAGS(X) X(69) X(95)', '#define VAR_FLAGS(X) X(69) X(95) X(325)')
 cuda+='''
std::vector<int64_t> epi_info(int64_t f,int64_t stages,int64_t M){cudaFuncAttributes a{};int r=0;int b=smem_bytes(1,stages,M);
#define INFO(F) if(f==F){C10_CUDA_CHECK(cudaFuncGetAttributes(&a,moe_var<1,F>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r,moe_var<1,F>,NTHR,b));}
 INFO(69) INFO(95) INFO(325)
#undef INFO
 return {r,a.numRegs,(int64_t)a.localSizeBytes,b,(int64_t)a.sharedSizeBytes};}
'''
 cell=cell.replace(repr(old),repr(cuda),1)
 cell=cell.replace('void run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);', 'void run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\nstd::vector<int64_t> epi_info(int64_t,int64_t,int64_t);')
 cell=cell.replace('SMALLM_FLAGS_BUILT = [69, 95]', 'SMALLM_FLAGS_BUILT = [69, 95, 325]')
 cell=cell.replace("'resident_blocks', 'run']", "'resident_blocks', 'run', 'epi_info']")
 cell=cell.replace('flags, stages=2, grid=264):', 'flags, stages=2, grid=264, stamps=None):')
 cell=cell.replace("state['sms'] * mod.resident_blocks(1, stages, m)", "state['sms'] * mod.epi_info(flags, stages, m)[0]")
 cell=cell.replace('    mod.run(ptrs, m,', '    if flags & 256:\n        if stamps is None or stamps.shape != (grid,64):raise ValueError("epilogue trace needs [grid,64]")\n        ptrs.append(stamps.data_ptr())\n    mod.run(ptrs, m,')
 # Standalone cells cannot assume the caller imported hashlib/torch/load_inline already.
 prefix=('epib','epiv','epip','epin')[mode]
 cell='import hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'+cell
 cell=cell.replace('SMALLM_VAR_',prefix.upper()+'_').replace('SMALLM_FLAGS_BUILT',prefix.upper()+'_FLAGS_BUILT').replace('_SMALLM_MODULE','_'+prefix.upper()+'_MODULE').replace('smallm_',prefix+'_')
 return cell,cuda

def build():
 for mode in range(4):
  cell,cuda=source(mode);tag=('epib','epiv','epip','epin')[mode]
  for p,s in ((DEV/(tag+'_cell.py'),cell),(DEV/(tag+'.cu'),cuda)):
   p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
