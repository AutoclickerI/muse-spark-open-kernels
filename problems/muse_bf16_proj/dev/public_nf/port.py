"""Layer adapter for Nathen Fernandes' MIT kernels at 86aff75.
Adaptations: FP32 layer outputs, per-row-tile exact coding origins, S14 PDL,
N4608 pool_router, N1280 qkv3. Original copyright/permission in nf_selected.py.
"""
import hashlib,re
import torch
from torch.utils.cpp_extension import load_inline

# Populated by build_port.py with an isolated attributed vendor namespace.
NF_VENDOR = None


def _port_cuda(src, *, dynamic_rows=False, m32w=False):
    """Apply narrow, asserted changes to reviewed upstream CUDA."""
    if m32w:
        assert '__nv_bfloat16* out,int m,int n,int k' in src
        src=src.replace('__nv_bfloat16* out,int m,int n,int k','float* out,int m,int n,int k')
        src=src.replace('(__nv_bfloat16*)out,m,n,k','(float*)out,m,n,k')
        old='const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;'
        assert old in src
        src=src.replace(old,old.replace('*64;','*66;')+'\n    t0=el[64];t1=el[65];')
        old='__nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);\n    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);'
        assert old in src
        src=src.replace(old,'*(float4*)(out+(long)(mt*BN+token)*n+nt*R+row)=sum;')
    else:
        old='using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;'
        assert old in src
        src=src.replace(old,'using OutT = float;')
        needle='const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;'
        assert needle in src
        src=src.replace(needle,needle+'\n    T0=esc[(size_t)gw*34+32]; T1=esc[(size_t)gw*34+33];')
        assert 'esc + (size_t)gw * 32 + lane' in src
        src=src.replace('esc + (size_t)gw * 32 + lane','esc + (size_t)gw * 34 + lane')
        if dynamic_rows:
            # Only fixed template-instance CN fields; all addressing uses N/grid geometry.
            src=re.sub(r'(?<=,)\s*(2304|4096)(?=\s*,\s*8192)', '0',src)
            src=re.sub(r'(?<=, )2304(?=, 8192)', '0',src)
    # Layer producers release dependents at entry; X loads wait for their producer.
    # Gate GEMV is deliberately left in S14's specialized same-stream PDL kernel.
    marker='  extern __shared__'
    p=src.find(marker,src.find('__global__'))
    # Not every source's first helper is the selected producer; inject at kernel body
    # by its shared declaration (all selected warp / m32w kernels start here).
    assert p>=0
    src=src[:p]+'  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n'+src[p:]
    # Each source has one selected cudaLaunchKernelEx config (small families).
    if m32w:
        src=src.replace('cudaLaunchAttribute attr={};','cudaLaunchAttribute attr={};')
        # Existing attrs is cluster only. Give PDL a second independent attribute.
        old='if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}'
        assert old in src
        src=src.replace(old,'cudaLaunchAttribute attrs[2]; int na=0; if(split>1)attrs[na++]=attr; attrs[na].id=cudaLaunchAttributeProgrammaticStreamSerialization; attrs[na++].val.programmaticStreamSerializationAllowed=1; cfg.attrs=attrs;cfg.numAttrs=na;')
    else:
        # Existing generic launch allocates at[1]; append PDL after cluster setup.
        src=re.sub(r'cudaLaunchAttribute at\[(\d+)\];',lambda m:'cudaLaunchAttribute at['+str(int(m[1])+1)+'] = {};',src)
        needle='  cudaLaunchKernelEx(&lc, kern,'
        assert needle in src
        src=src.replace(needle,'  at[lc.numAttrs].id=cudaLaunchAttributeProgrammaticStreamSerialization;\n  at[lc.numAttrs].val.programmaticStreamSerializationAllowed=1;\n  lc.attrs=at; ++lc.numAttrs;\n'+needle)
    return src


def _nf_vendor():
    ns={}
    exec(compile(NF_VENDOR,'<NathenFernandes86aff75>', 'exec'),ns)
    for name in ('v18d_SM_CUDA_TEMPLATE','V18S_CUDA_TEMPLATE','tt_v10_sm2_SMALLM2_CUDA_TEMPLATE',
                 'tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE','tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE',
                 'dd_decodediet_CPSSM_CUDA_TEMPLATE'):
        ns[name]=_port_cuda(ns[name])
    for name in ('BUNDLE_QKV_CUDA','tt_v10_m32x__Q_CUDA','b2q_tt_v10_m32x__Q_CUDA','b2p_tt_v10_m32x__CUDA'):
        ns[name]=_port_cuda(ns[name],dynamic_rows=True)
    cpp,cuda,name=ns['mt_M16TRACE_SOURCES']['o']
    ns['mt_M16TRACE_SOURCES']['o']=(cpp,_port_cuda(cuda),name+'_layer')
    # C++ exported symbol must follow the extension binding name.
    old=ns['mt_M16TRACE_SOURCES']['o'][2]
    ns['mt_M16TRACE_SOURCES']['o']=(cpp.replace(name,old),ns['mt_M16TRACE_SOURCES']['o'][1].replace(name,old),old)
    return ns


# family, (WR,WK,KU,S), MT,ST,RED,NTMAX, extras
NF_SMALL={
 (1,'qkv_gate'):('d9',(1,4,8,4),1,2,2,256,()),
 (8,'qkv_gate'):('q8',(1,4,8,4),1,2,5,256,()),
 (16,'qkv_gate'):('q16',(1,4,8,4),2,1,5,128,()),
 (32,'qkv_gate'):('q32',(1,4,8,4),4,1,5,128,()),
 (1,'pool_router'):('v18',(2,16,8,1),1,1,0,1024,(6,)),
 (8,'pool_router'):('sm2',(2,16,8,1),1,1,0,1024,(6,)),
 (16,'pool_router'):('cp',(1,8,16,1),2,2,0,256,()),
 (32,'pool_router'):('p32',(1,8,16,1),4,1,0,256,()),
 (1,'o'):('sm2',(4,8,2,1),1,2,0,1024,(7,)),
 (8,'o'):('v18',(1,8,2,1),1,2,0,256,(6,)),
 (16,'o'):('o16',(2,4,4,1),2,1,0,256,()),
 (1,'unpooler_slice'):('u1',(1,8,2,4),1,2,2,256,()),
 (8,'unpooler_slice'):('u8',(1,16,2,2),1,2,2,512,()),
 (16,'unpooler'):('cp',(2,4,16,1),2,2,0,256,()),
}
for _t in (1,16,32):
    NF_SMALL[_t,'qkv3']=NF_SMALL[_t,'qkv_gate']


def _nf_group_pack(ns,w,name,family):
    prefix={'d9':'v18d_v18d_native','v18':'v18s','sm2':'tt_v10_sm2_smallm2',
        'u1':'tt_v12_smtrace1_smtrace','u8':'tt_v13_smtrace8_smtrace','cp':'dd_decodediet_dd_cap2_b_cpssm'}
    if family in prefix:
        pack=ns[prefix[family]+'_pack_format']
        stream=ns[prefix[family]+'_stream']
    elif family=='o16':pack=lambda w,fmt:ns['mt_wp0_pack_format'](w,fmt,0,0);stream=ns['mt_wp0_stream']
    else:
        pack=lambda w,fmt:ns['tt_v10_m32x_m32x_pack_format'](w,fmt,0,5 if family.startswith('q') else 0)
        # p32 is D9 and uses the b2 packer, not the older decoder expression.
        if family=='p32':pack=lambda w,fmt:ns['b2p_tt_v10_m32x_m32x_pack_format'](w,fmt,0,5)
        stream=ns['tt_v10_m32x_m32x_stream']
    groups=(w[:4096],w[4096:4352],w[4352:]) if name=='pool_router' else (w,)
    parts=[];tables=[];es=[];offset=0
    for group in groups:
        pk=pack(group.contiguous(),1)
        if pk is None:raise RuntimeError('NF lossless pack rejected '+name)
        parts.append(pk['data']);tab=pk['t01']&((1<<64)-1)
        tables.extend([(tab&0xffffffff,tab>>32)]*(group.shape[0]//16))
        ee=list(pk['esc']);ee[0]=ee[0]+offset;es.append(ee);offset+=group.shape[0]//16
    pk=dict(pk,data=torch.cat(parts,0).contiguous(),esc=tuple(torch.cat([e[i] for e in es]) for i in range(len(es[0]))))
    return pk,torch.tensor(tables,device=w.device,dtype=torch.int64),stream


def nf_prepare(weights,keys=None):
    requested=set(NF_SMALL)|NF_WIDE if keys is None else set(keys)
    ns=_nf_vendor();dev=weights['qkv_gate'].device
    hi=weights['router'].to(torch.bfloat16)
    roles={'qkv_gate':weights['qkv_gate'],'qkv3':weights['qkv_gate'][:1280].contiguous(),
           'o':weights['o'],'pool_router':torch.cat([weights['pooler'],hi,(weights['router']-hi.float()).to(torch.bfloat16)]),
           'unpooler_slice':weights['unpooler'][:1024].contiguous(),'unpooler':weights['unpooler']}
    native=torch.cuda.get_device_capability(dev)==(9,0)
    state={'ns':ns,'roles':roles,'plans':{},'native':native,'modules':{},'requested':requested}
    grouped={};prepared={}
    for key,(family,cfg,mt,st,red,ntmax,extra) in NF_SMALL.items():
        if key not in requested:continue
        t,name=key;w=roles[name];n,k=w.shape
        inst=(1,mt,1,st,red,0,ntmax,0,n,k,*cfg,*extra)
        grouped.setdefault(family,[]).append((key,inst,cfg))
    for family,entries in grouped.items():
        insts=list(dict.fromkeys(e[1] for e in entries))
        # Cluster instructions cannot run on local sm_120; force their Python ABI
        # through exact-format arithmetic below. Noncluster warp MMA runs natively.
        if not native and any(e[2][-1]>1 for e in entries):continue
        if family=='d9':mod=ns['v18d_v18d_native_build'](insts,'nf_layer_d9');symbol='v18d_native_launch'
        elif family=='v18':mod=ns['v18s_build'](insts,'nf_layer_v18');symbol='v18s_launch'
        elif family=='sm2':mod=ns['tt_v10_sm2_smallm2_build'](insts,'nf_layer_sm2');symbol='smallm2_launch'
        elif family in ('u1','u8'):
            pref='tt_v12_smtrace1' if family=='u1' else 'tt_v13_smtrace8'
            mod=ns[pref+'_smtrace_build'](insts,'nf_layer_'+family);symbol='smtrace_launch'
        elif family=='cp':mod=ns['dd_decodediet_cps_build']('sm',insts,2);symbol='dd_cap2_b_cpssm_launch'
        elif family in ('q16','q32','p32'):
            pref={'q16':'b2q_tt_v10_m32x__q','q32':'tt_v10_m32x__q','p32':'b2p_tt_v10_m32x_'}[family]
            mod=ns[pref+'_build']()
            symbol={'q16':'r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch','q32':'m32x_q_launch','p32':'r3_pool32_pooler_d9_cap2_a_m32x_launch'}[family]
        elif family=='q8':
            symbol='qs_literal_owner5_decode5_launch';cpp=ns['SM_CPP'].replace('sm_launch',symbol)
            mod=load_inline(name='nf_layer_q8',cpp_sources=cpp,cuda_sources=ns['BUNDLE_QKV_CUDA'],functions=[symbol],extra_cuda_cflags=['-O3','-lineinfo',ns['_nf_arch_flag']()],no_implicit_headers=True,with_pytorch_error_handling=False)
        elif family=='o16':
            cpp,cuda,symbol=ns['mt_M16TRACE_SOURCES']['o']
            mod=load_inline(name='nf_layer_o16',cpp_sources=cpp,cuda_sources=cuda,functions=[symbol],extra_cuda_cflags=['-O3','-lineinfo',ns['_nf_arch_flag']()],no_implicit_headers=True,with_pytorch_error_handling=False)
        else:raise RuntimeError(family)
        state['modules'][family]=mod
        for key,inst,cfg in entries:
            name=key[1];w=roles[name];pk,tables,stream=_nf_group_pack(ns,w,name,family)
            sp=stream(pk,1,*cfg)
            if sp is None:raise RuntimeError('NF escape capacity '+str(key))
            # One independent high-byte origin/table per row tile. RT is1 in selected paths.
            wr,wk,ku,s=cfg;gx=w.shape[0]//(wr*16);G=gx*s*wr*wk
            gw=torch.arange(G,device=dev);tile=((gw//(wr*wk))%gx)*wr+(gw%wr)
            meta=torch.empty((G,34),device=dev,dtype=torch.int32);meta[:,:32]=sp['esc']
            meta[:,32:]=tables[tile].to(torch.int32)
            idx=0 if family in ('q16','q32','p32','o16') else 1 if family=='q8' else insts.index(inst)
            state['plans'][key]=dict(sp,esc=meta,fn=getattr(mod,symbol),idx=idx,cfg=cfg,n=w.shape[0])
    return state


def nf_project(x,name,state):
    """Return FP32 S=1 matrix; forced local cluster paths use an explicit stand-in."""
    key=x.shape[0],name;p=state['plans'].get(key)
    if p is None:
        if state['native']:raise RuntimeError('missing NF plan '+str(key))
        return torch.mm(x,state['roles'][name].T,out_dtype=torch.float32).unsqueeze(0),1
    out=torch.empty((x.shape[0],p['n']),device=x.device,dtype=torch.float32)
    args=(p['idx'],p['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),p['esc'].data_ptr(),x.shape[0],p['n'],x.shape[1],*p['cfg'],p['t01'],p['u01'],torch.cuda.current_stream(x.device).cuda_stream,0)
    rc=p['fn'](*args)
    if rc:raise RuntimeError('NF launch '+str(key)+' rc='+str(rc))
    return out.unsqueeze(0),1

NF_WIDE={(32,'o'),(32,'unpooler')} | {(t,n) for t in (128,256) for n in ('qkv_gate','o','pool_router','unpooler')}


def _nf_pdl_wide(src):
    # One selected producer per source. Add PDL at the launch and wait before TMA reads X.
    p=src.index('  extern __shared__',src.index('__global__'))
    src=src[:p]+'  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n'+src[p:]
    src=src.replace('cudaLaunchKernelEx(', 'nf_port_launch(')
    helper='''#include <cuda_runtime.h>
template<typename F,typename...A> static cudaError_t nf_port_launch(cudaLaunchConfig_t* c,F f,A...a) {
  cudaLaunchAttribute at[8];int n=c->numAttrs;
  for(int i=0;i<n;++i)at[i]=c->attrs[i];
  at[n].id=cudaLaunchAttributeProgrammaticStreamSerialization;at[n].val.programmaticStreamSerializationAllowed=1;
  c->attrs=at;c->numAttrs=n+1;return cudaLaunchKernelEx(c,f,a...);
}
'''
    return helper+src


def _nf_prepare_wide(state,weights):
    if not state['native'] or not state['requested']&NF_WIDE:return
    ns=state['ns'];dev=weights['o'].device;requested=state['requested']
    # Register/shared WGMMA: preserve FP32 before existing norm/route finalizers.
    if requested & {(32,'unpooler'),(128,'pool_router'),(256,'pool_router')}:
        insts=[(32,64,4,2,2,True,1),(64,64,4,2,2,True,1)]
        cuda=_port_cuda(ns['dd_decodediet_tt_mw_cuda_source'](insts),m32w=True)
        tag=hashlib.sha256(cuda.encode()).hexdigest()[:12]
        mod=load_inline(name='nf_layer_mw_'+tag,cpp_sources=ns['dd_decodediet_tt_mw_CPP'],cuda_sources=cuda,functions=['launch','info'],
            extra_cuda_cflags=['-O3','-std=c++20',ns['_nf_arch_flag']()],no_implicit_headers=True,with_pytorch_error_handling=False)
        for t,name,idx in ((32,'unpooler',0),(128,'pool_router',1),(256,'pool_router',1)):
            if (t,name) not in requested:continue
            w=state['roles'][name];groups=(w[:4096],w[4096:4352],w[4352:]) if name=='pool_router' else (w,)
            ps=[];metas=[]
            for group in groups:
                pk=ns['dd_decodediet_tt_mw_pack_weight'](group.contiguous(),64,4,1,True)
                if pk is None:raise RuntimeError('NF m32w pack '+name)
                G=pk['esc'].shape[0];meta=torch.empty((G,66),device=dev,dtype=torch.int32);meta[:,:64]=pk['esc']
                tab=pk['table']&((1<<64)-1)
                meta[:,64]=(tab&0xffffffff)-(1<<32) if tab&0x80000000 else tab&0xffffffff
                top=tab>>32;meta[:,65]=top-(1<<32) if top&0x80000000 else top
                ps.append(pk['w']);metas.append(meta)
            state['plans'][t,name]=dict(kind='mw',fn=mod.launch,idx=idx,wp=torch.cat(ps),esc=torch.cat(metas),n=w.shape[0])
    # Raw BF16 SS WGMMA O shapes, with original selected tiling and output transpose/store.
    ns['b2o_OLAYER_CUDA']=_nf_pdl_wide(ns['b2o_OLAYER_CUDA'])
    ns['OLAYER_CUDA']=_nf_pdl_wide(ns['OLAYER_CUDA'])
    for t,build,run,cfg in ((32,'b2o_prepare_olayer_candidates','b2o_run_olayer_candidate',(32,128,8,1,1,1)),
                           (128,'prepare_olayer_candidates','run_olayer_candidate',(128,128,4,3,1,1))):
        if (t,'o') not in requested:continue
        st=ns[build](weights,[cfg]);state['plans'][t,'o']=dict(kind='ol',run=ns[run],state=st)
    if (256,'o') in requested:
        ns['tt_v10_tune_RL_CUDA']=_nf_pdl_wide(ns['tt_v10_tune_RL_CUDA'])
        mod=ns['tt_v10_tune_build_rl']([(256,64,5,7,1,1,1)])
        state['plans'][256,'o']=dict(kind='rl',mod=mod,w=ns['tt_v10_tune_pack_weight'](weights['o'],64))
    if (128,'qkv_gate') in requested:
        ns['tt_v14_qkv_QKVBIG_CUDA']=_nf_pdl_wide(ns['tt_v14_qkv_QKVBIG_CUDA'])
        mod=ns['tt_v14_qkv_build_qkv']([(64,36,128,3,8,539,5)])
        state['plans'][128,'qkv_gate']=dict(kind='qb',mod=mod,w=ns['tt_v14_qkv_qbig_pack'](weights['qkv_gate'],36))
    # Exactly pinned cuBLASLt descriptors, same original matrix precision.
    if requested & {(256,'qkv_gate'),(128,'unpooler'),(256,'unpooler')}:
        lt=ns['tt_lt__lt_state'](weights)
        for key in ((256,'qkv_gate'),(128,'unpooler'),(256,'unpooler')):
            if key not in requested:continue
            layout,epilogue,desc=ns['tt_lt__LT_H200_FIXED'][key]
            plan=ns['tt_lt__lt_plan'](lt,key[1],key[0],layout,epilogue);idx=plan.restore(desc)
            if idx<0:raise RuntimeError('NF LT descriptor rejected '+str(key))
            state['plans'][key]=dict(kind='lt',plan=plan,idx=idx)
        state['lt']=lt


_nf_small_prepare=nf_prepare
_nf_small_project=nf_project

def nf_prepare(weights,keys=None):
    state=_nf_small_prepare(weights,keys)
    _nf_prepare_wide(state,weights)
    # Warm both native launch attributes and local arithmetic stand-ins before graph capture.
    for t,name in state['requested']:
        x=torch.zeros((t,state['roles'][name].shape[1]),device=state['roles'][name].device,dtype=torch.bfloat16)
        nf_project(x,name,state)
    state['plans']={k:v for k,v in state['plans'].items() if k in state['requested']}
    return state


def nf_project(x,name,state):
    p=state['plans'].get((x.shape[0],name))
    if p is None or 'kind' not in p:return _nf_small_project(x,name,state)
    kind=p['kind'];n=state['roles'][name].shape[0];stream=torch.cuda.current_stream(x.device).cuda_stream
    if kind=='mw':
        out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.float32)
        rc=p['fn'](p['idx'],x.data_ptr(),p['wp'].data_ptr(),p['esc'].data_ptr(),out.data_ptr(),x.shape[0],n,x.shape[1],1,0,stream)
    elif kind=='ol':return p['run'](x,p['state'],0).unsqueeze(0),1
    elif kind=='lt':
        out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.bfloat16);rc=p['plan'].run(x,out,p['idx'])
    elif kind in ('rl','qb'):
        out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.bfloat16)
        rc=p['mod'].rl_launch(0,x.data_ptr(),p['w'].data_ptr(),out.data_ptr(),x.shape[0],stream,0) if kind=='rl' else p['mod'].qb_launch(0,p['w'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0,0)
    else:raise RuntimeError(kind)
    if rc:raise RuntimeError('NF wide '+kind+' rc='+str(rc))
    return out.unsqueeze(0),1
