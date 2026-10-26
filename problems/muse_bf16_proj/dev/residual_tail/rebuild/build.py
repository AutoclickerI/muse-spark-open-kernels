"""One mechanism change from measured N: a dedicated b2 tail-CTA path."""
import ast
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3]
BASE='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'

def main():
    base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()==BASE
    native=(P/'native_generated.py.inc').read_text()
    native=native[:native.index('template<bool POST,int MODE>void rt_launch')]
    anchor='  float4 a[G], b[G], w[G], q[G];'
    assert native.count(anchor)==1
    tail='''
  if constexpr(POST){
    if(rank>0){
      static_assert(C==8 && NT==256 && ROUNDS==1 && MODE==2);
      const int j=(rank*NT+tid)*4;
      // The last seven tiles contain no unpooler branch. Only these two
      // immutable vectors are needed; no beta2 or post-FFN weights. Explicit
      // preloads also reduce this specialization from 114 to 98 registers.
      float4 tw=rt_load4(prew+j),ta=rt_load4(alpha+j);
      pdl_wait();
      float4 ts=rt_load4(stream+(long long)t*HID+j);
      ts.x*=ta.x;ts.y*=ta.y;ts.z*=ta.z;ts.w*=ta.w;
      float v[4]={0.f,rt_load1(stats+(long long)t*C+rank),0.f,0.f};
      // Same DSM publication and cluster barrier as the original prefix CTA.
      cluster_sum<4,C>(cl,v,slots[0]);
      float inv_br=rsqrtf(v[0]/HID+1e-8f);
      float inv=rsqrtf((v[1]+2.f*inv_br*v[2]+inv_br*inv_br*v[3])/HID+1e-5f);
      *reinterpret_cast<float4*>(sout+(long long)t*HID+j)=ts;
      __nv_bfloat162 o0=__floats2bfloat162_rn(ts.x*inv*tw.x,ts.y*inv*tw.y);
      __nv_bfloat162 o1=__floats2bfloat162_rn(ts.z*inv*tw.z,ts.w*inv*tw.w);
      *reinterpret_cast<uint2*>(out+(long long)t*HID+j)=make_uint2(*reinterpret_cast<uint32_t*>(&o0),*reinterpret_cast<uint32_t*>(&o1));
      return;
    }
  }
'''
    native=native.replace(anchor,tail+anchor).replace('namespace glue_cl {','namespace rtf_cl {')
    launch=(P/'launch.cu.inc').read_text()
    launch=launch[:launch.index('void rt_boundary')]+'''
void rt_boundary(std::vector<int64_t>v,int64_t T,int64_t post,int64_t mode,int64_t stream){
 if(post!=1 || mode!=2)throw std::runtime_error("fast tail only post mode2");
 rt_launch<true,2>(v,T,(cudaStream_t)stream);
}
std::vector<int64_t> rt_info(int64_t post,int64_t mode){
 cudaFuncAttributes a{};cudaFuncGetAttributes(&a,rtf_cl::boundary_tail<8,256,1,true,2>);
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes};
}
'''
    native+=launch.replace('glue_cl::','rtf_cl::')
    runtime=(P/'runtime_generated.py.inc').read_text()
    module_source=(P/'runtime.py.inc').read_text().split('_rt_original_boundary=')[0]
    module_source=module_source.replace('_RT_MODULES','_RTF_MODULES').replace('rt_module','rt_fast_module').replace('RT_CUDA','RT_FAST_CUDA').replace("name='p7_rt_'","name='p7_rtf_'")
    runtime+='\nRT_FAST_CUDA='+repr(native)+'\n'+module_source
    # Original N and T use the identical original N b1 function/handle. Only
    # the T b2 launch is redirected into this new native kernel.
    runtime=runtime.replace("'C':3}[RT_ARM]","'C':3,'T':2}[RT_ARM]")
    anchor='    rt_module().rt_boundary(args,t,post,mode,torch.cuda.current_stream().cuda_stream)'
    assert runtime.count(anchor)==1
    runtime=runtime.replace(anchor,"    mod=rt_fast_module() if post and RT_ARM=='T' else rt_module()\n    mod.rt_boundary(args,t,post,mode,torch.cuda.current_stream().cuda_stream)")
    runtime+='''
_rt_rebuild_prepare=prepare
def prepare(weights):
    state=_rt_rebuild_prepare(weights);rt_fast_module();return state
'''
    compile(runtime,'p7-rebuild-runtime','exec')
    (H/'native_generated.py.inc').write_text(native)
    (H/'runtime_generated.py.inc').write_text(runtime)
    code=base+'\n'+runtime+'\nRT_ARM="T"\n'
    (H/'T_trial.py').write_text(code)
    manifest=dict(base_sha256=BASE,native_sha256=hashlib.sha256(native.encode()).hexdigest(),
                  runtime_sha256=hashlib.sha256(runtime.encode()).hexdigest(),
                  candidate_sha256=hashlib.sha256(code.encode()).hexdigest(),arms=['A','A2','N','T'],
                  original_N_native_sha256=json.loads((P/'manifest.json').read_text())['native_sha256'])
    (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('T dedicated tail',manifest['native_sha256'][:8],manifest['candidate_sha256'][:8])

if __name__=='__main__':main()
