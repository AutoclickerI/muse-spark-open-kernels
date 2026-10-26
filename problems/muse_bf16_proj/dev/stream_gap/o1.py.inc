SG_O1_ARM=0

def sgo_source():
    ns=_nf_vendor();src=ns['tt_v10_sm2_SMALLM2_CUDA_TEMPLATE'];src=src[:src.index('static int smallm2_smem_optin')]
    signature='uint32_t U0, uint32_t U1) {';assert signature in src
    i=src.index(signature,src.index('__global__'));src=src[:i]+src[i:].replace(signature,'uint32_t U0, uint32_t U1, SGOArgs fused) {',1)
    p=src.index('  extern __shared__',src.index('__global__'))
    src=src[:p]+'  __shared__ int sgo_last;\n  __shared__ float sgo_scratch[32];\n'+src[p:]
    i=src.rfind('\n}')
    src=src[:i]+'''
  if(fused.mode) {
    // Every writing thread publishes before the completion ticket. CG loads
    // in the final CTA bypass a stale private L1 from an earlier replay.
    __threadfence();__syncthreads();
    if(threadIdx.x==0)sgo_last=atomicAdd(fused.counter,1u)==gridDim.x-1;
    __syncthreads();
    if(sgo_last) {__threadfence();sgo_finish<NTMAX>((const float*)outp,fused,sgo_scratch);}
  }
'''+src[i:]
    host='''
template<int WR> int sgo_launch_t(int64_t wp,int64_t esc,int64_t x,int64_t out,int64_t stream_in,int64_t alpha,int64_t beta,int64_t prew,int64_t hidden,int64_t sout,int64_t counter,int mode,int64_t stream) {
 constexpr int NT=WR*8*32,SM=WR*8*512;
 auto fn=smallm2_gemv<1,1,1,2,0,0,NT,0,8192,1024,WR,8,2,1,7>;
 SGOArgs args{(const float*)stream_in,(const float*)alpha,(const float*)beta,(const float*)prew,(float*)sout,(__nv_bfloat16*)hidden,(uint32_t*)counter,mode};
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(8192/(WR*16),1,1);cfg.blockDim=dim3(NT,1,1);cfg.dynamicSmemBytes=SM;cfg.stream=(cudaStream_t)stream;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&attr;cfg.numAttrs=1;
 auto e=cudaLaunchKernelEx(&cfg,fn,(const int4*)wp,(const __nv_bfloat16*)x,(void*)out,(const uint32_t*)esc,1,8192,1024,WR,8,2,0u,0u,0u,0u,args);
 return e?int(e):int(cudaGetLastError());
}
int sgo_launch(int wr,int64_t wp,int64_t esc,int64_t x,int64_t out,int64_t stream_in,int64_t alpha,int64_t beta,int64_t prew,int64_t hidden,int64_t sout,int64_t counter,int mode,int64_t stream) {
 switch(wr) {case 4:return sgo_launch_t<4>(wp,esc,x,out,stream_in,alpha,beta,prew,hidden,sout,counter,mode,stream);case 2:return sgo_launch_t<2>(wp,esc,x,out,stream_in,alpha,beta,prew,hidden,sout,counter,mode,stream);case 1:return sgo_launch_t<1>(wp,esc,x,out,stream_in,alpha,beta,prew,hidden,sout,counter,mode,stream);default:return -99;}
}
'''
    return SGO_HELPER+'\n'+src+'\n'+host

def sgo_prepare(weights):
    src=sgo_source();ns=_nf_vendor();tag=hashlib.sha256(src.encode()).hexdigest()[:12]
    cpp='#include <pybind11/pybind11.h>\n#include <stdint.h>\nint sgo_launch(int,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int64_t);\n'
    mod=load_inline(name='sgo_fused_'+tag,cpp_sources=cpp,cuda_sources=src,functions=['sgo_launch'],extra_cuda_cflags=['-O3','-std=c++20',ns['_nf_arch_flag']()],no_implicit_headers=True,with_pytorch_error_handling=False)
    w=weights['o'];pk,tables,stream=_nf_group_pack(ns,w,'o','sm2');plans={}
    for wr in (4,2,1):
        cfg=(wr,8,2,1);sp=stream(pk,1,*cfg);assert sp is not None
        g=8192//(wr*16)*wr*8;gw=torch.arange(g,device=w.device);gx=8192//(wr*16);tile=(gw//(wr*8)%gx)*wr+gw%wr
        meta=torch.empty((g,34),device=w.device,dtype=torch.int32);meta[:,:32]=sp['esc'];meta[:,32:]=tables[tile].to(torch.int32)
        plans[wr]=dict(sp,esc=meta,counter=torch.zeros(1,device=w.device,dtype=torch.int32))
    return dict(mod=mod,plans=plans)

def sgo_project(x,residual,state,wr,mode=1):
    p=state['sgo']['plans'][wr];out=torch.empty((1,8192),device=x.device,dtype=torch.float32);hidden=torch.empty((1,8192),device=x.device,dtype=torch.bfloat16);sout=torch.empty_like(out)
    rc=state['sgo']['mod'].sgo_launch(wr,p['wp'].data_ptr(),p['esc'].data_ptr(),x.data_ptr(),out.data_ptr(),residual.data_ptr(),state['alpha1'].data_ptr(),state['beta1'].data_ptr(),state['pre_ffn_w'].data_ptr(),hidden.data_ptr(),sout.data_ptr(),p['counter'].data_ptr(),mode,torch.cuda.current_stream().cuda_stream)
    if rc:raise RuntimeError('sgo launch '+str((wr,mode,rc)))
    return (out.unsqueeze(0),1),(hidden,sout)

_sgo_old_prepare=prepare
_sgo_old_run=_best7_run
_sgo_old_project=_best7_project
_sgo_old_boundary=_best7_boundary

def prepare(weights):
    global SG_O1_ARM
    selected=SG_O1_ARM;SG_O1_ARM=0
    try:st=_sgo_old_prepare(weights)
    finally:SG_O1_ARM=selected
    st['sgo']=sgo_prepare(weights);return st

def _best7_run(data,state,plan,prefetch=None):
    active=SG_O1_ARM and data['config']['tokens']==1
    if active:state['_sgo_inputs']=data['stream']
    try:return _sgo_old_run(data,state,plan,prefetch)
    finally:
        state.pop('_sgo_inputs',None);state.pop('_sgo_result',None)

def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    if name=='o' and '_sgo_inputs' in state:
        pair,result=sgo_project(x,state['_sgo_inputs'],state,SG_O1_ARM)
        state['_sgo_result']=result;return pair
    return _sgo_old_project(x,name,state,plan,out_dtype)

def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    if phase=='b1' and '_sgo_result' in state:return state['_sgo_result']
    return _sgo_old_boundary(branch,stream,state,plan,phase,sync)
