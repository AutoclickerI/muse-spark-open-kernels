"""Raw/recoded register-A WGMMA plus the stock NT1 recoded control, with pointer bindings."""
import ast,re,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3]
sys.path.insert(0,str(P));from native_abi import PRELUDE,lean

def main():
 s=(P/'w2fix.cu').read_text().replace('namespace mwfx','namespace mwrc').replace('mwfx::','mwrc::')
 s=re.sub(r'\bfx_', 'rc_',s).replace('FX_CASES','RC_CASES')
 s=s.replace('int MODE>','int MODE,bool RECODED>')
 s=s.replace('__device__ __forceinline__ void dq(', 'template<bool RECODED>__device__ __forceinline__ void dq(')
 a=s.index(' unsigned s0=lut');b=s.index('\n out[0]=',a)
 s=s[:a]+''' unsigned s0,s1,s2,s3;
 if constexpr(RECODED){s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u;s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u;s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;}
 else{s0=lut[raw&255];s1=lut[(raw>>8)&255];s2=lut[(raw>>16)&255];s3=lut[raw>>24];}'''+s[b:]
 s=s.replace('dq(q.x,q.y,sf,lut,','dq<RECODED>(q.x,q.y,sf,lut,').replace('__shared__ unsigned lut[256]','__shared__ unsigned lut[RECODED?1:256]')
 s=s.replace(' lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();',' if constexpr(!RECODED){lut[t]=sf_bits(t);lut[t+128]=sf_bits(t+128);__syncthreads();}')
 a=s.index('int64_t rc_init()');b=s.index('void rc_route(',a)
 s=s[:a]+'''int64_t rc_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,bool R>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwrc::fused<NT,KT,S,SK,F,1,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,false>();set.template operator()<false,false>();set.template operator()<true,true>();set.template operator()<false,true>();}
 RC_CASES(INIT)
 #undef INIT
 return opt;
}
std::vector<int64_t> rc_info(int64_t which,int64_t first,int64_t encoded){
 #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,bool R>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwrc::fused<NT,KT,S,SK,F,1,R>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwrc::fused<NT,KT,S,SK,F,1,R>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return encoded?info.template operator()<true,true>():info.template operator()<true,false>();return encoded?info.template operator()<false,true>():info.template operator()<false,false>();}
 switch(which){RC_CASES(INFO) default:WX_CHECK(false,"unknown recoded configuration");}
 #undef INFO
}
'''+s[b:]
 a=s.index('void rc_run(');b=s.index('namespace mwrc {\n__global__ void finish',a)
 host=s[a:b].replace('int64_t mode,','int64_t encoded,').replace('mwrc::fused<NT,KT,S,SK,FIRST,MODE>','mwrc::fused<NT,KT,S,SK,FIRST,1,RECODED>').replace('<bool FIRST,int MODE,bool RECODED>','<bool FIRST,bool RECODED>')
 x=host.index('};if(first)');y=host.index('break;}',x)
 host=host[:x]+'''};if(first){if(encoded)launch.template operator()<true,true>();else launch.template operator()<true,false>();}else{if(encoded)launch.template operator()<false,true>();else launch.template operator()<false,false>();}'''+host[y:]
 s=s[:a]+host+s[b:];s=s.replace('// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.','// Two lossless banks: raw E4M3 and exponent/mantissa bytes with sign/zero folded into FP4.\n// Both convert to BF16 only inside the timed kernel.')
 (H/'recode.cu').write_text(s)
 t=ast.parse((R/'problems/muse_decode_layer/cand_r3_v9.py').read_text());v=next(n.value.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='V9_CUDA_SRC' for a in n.targets))
 v=lean(v[:v.index('int64_t max_smem_attr()')]).replace('namespace {','namespace n1 {',1)
 v+=r'''
int64_t n1_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));WX_CUDA_CHECK(cudaFuncSetAttribute(n1::moe_fused<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt));return opt;}
int64_t n1_smem(int64_t stages,int64_t M){return n1::smem_bytes(1,stages,M);}
int64_t n1_resident(int64_t stages,int64_t M){int n;WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,n1::moe_fused<1>,n1::NTHR,n1::smem_bytes(1,stages,M)));return n;}
void n1_run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream){
 WX_CHECK(v.size()==13);n1::moe_fused<1><<<grid,n1::NTHR,n1::smem_bytes(1,stages,M),(cudaStream_t)stream>>>((__nv_bfloat16*)v[0],(int*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(uint8_t*)v[4],(uint8_t*)v[5],(float*)v[6],(float*)v[7],(__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages);WX_CUDA_CHECK(cudaGetLastError());}
'''
 (H/'nt1.cu').write_text(PRELUDE+v)
 # Keep the organizer-proven encoder/repacker, including its precise row/K mapping.
 names={'H','NLOC','NT1','NT2','TOPK','TILE_BYTES','NCNT','_bit_table','_TABLE','_LA_MASK','_DQ_SLOTS','_encode_scales','_repack'}
 nodes=[n for n in t.body if (isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id in names for a in n.targets))]
 mapping={n.name:'n1'+n.name for n in nodes if isinstance(n,ast.FunctionDef)}|{a.id:'n1_'+a.id for n in nodes if isinstance(n,ast.Assign) for a in n.targets if isinstance(a,ast.Name)}
 class Rename(ast.NodeTransformer):
  def visit_Name(self,n):
   if n.id in mapping:n.id=mapping[n.id]
   return n
  def visit_FunctionDef(self,n):
   n=self.generic_visit(n);n.name=mapping.get(n.name,n.name);return n
 tree=Rename().visit(ast.Module(body=nodes,type_ignores=[]));pack=ast.unparse(ast.fix_missing_locations(tree));pack=pack.replace('    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and int(mant.max()) <= 7','    if bool(nz.any()):assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and int(mant.max()) <= 7');(H/'nt1_pack.py').write_text('import torch\n'+pack+'\n')
 print('generated',len(s),len(v))

if __name__=='__main__':main()
