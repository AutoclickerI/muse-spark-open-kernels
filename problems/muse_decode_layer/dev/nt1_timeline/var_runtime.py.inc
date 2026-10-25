VAR_CPP='''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
int64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);int64_t resident_blocks(int64_t,int64_t,int64_t);
void run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
'''
F_Z,F_SLOT,F_ZOUT,F_RSQ,F_NREG,F_PDL,F_LAST,F_TRACE,F_NT2,F_W16,F_RND,F_ROT,F_PIPE,F_WPUB,F_ALIGN=1,2,4,8,16,64,128,256,2048,4096,8192,16384,32768,65536,131072
VAR_MODULE=[]


def var_module():
    if not VAR_MODULE:
        tag=hashlib.sha256(VAR_CUDA.encode()).hexdigest()[:12]
        mod=load_inline(name='muse_smallm_var_'+tag,cpp_sources=VAR_CPP,cuda_sources=VAR_CUDA,
            functions=['max_smem_attr','smem_needed','resident_blocks','run'],no_implicit_headers=True,
            with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',
            *(['-gencode=arch=compute_90a,code=sm_90a'] if NT1_HOPPER else [])],verbose=False)
        mod.smem_optin=mod.max_smem_attr();VAR_MODULE.append(mod)
    return VAR_MODULE[0]


def var_geometry(state,flags,m,grid=264,stages=None):
    """(nt, stages, grid): NT2 runs 1 CTA/SM with as many stages as fit (4 on the H200's 227 KB; 2, or 1 for W16, on sm_120's 99 KB)."""
    mod=var_module();nt=2 if flags&F_NT2 else 1;f=flags&~F_TRACE
    if stages is None:stages=next((s for s in ((4,3,2,1) if nt==2 else (2,)) if mod.smem_needed(f,s,m)<=mod.smem_optin),2)
    if mod.smem_needed(f,stages,m)>mod.smem_optin:raise ValueError('stage depth exceeds shared capacity')
    return nt,stages,min(grid,state['sms']*mod.resident_blocks(f,stages,m))


def var_call(d,state,flags,grid=264,stages=None,stamps=None):
    """The flag-templated NT1 variant; flags=0 is the incumbent's math in the variant harness."""
    x=d['hidden_states'].contiguous();m=len(x);dev=x.device;mod=var_module()
    if flags&F_SLOT and m!=1:raise ValueError('F_SLOT is the M=1 dispatch')
    nt,stages,grid=var_geometry(state,flags,m,grid,stages)
    inter=torch.empty((m*8,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((m*8,4096),device=dev);ss=torch.empty(m*8,device=dev)
    out=torch.empty_like(x);ids=d['topk_ids'].int().contiguous();tw=d['topk_weights'].float().contiguous()
    cnt=state.setdefault('vcounters',{}).setdefault(flags&~F_TRACE,None)
    if cnt is None:cnt=state['vcounters'][flags&~F_TRACE]=torch.zeros(64,device=dev,dtype=torch.int32)
    tensors=[x,ids,tw,state['pnw'],state['bank']['w13p'],state['bank']['w2p'],state['gs13'],state['gs2'],inter,y,ss,cnt,out]
    ptrs=[v.data_ptr() for v in tensors]
    if flags&F_TRACE:
        if stamps is None or stamps.shape!=(grid,64):raise ValueError('F_TRACE needs a (grid,64) int64 stamp buffer')
        ptrs.append(stamps.data_ptr())
    mod.run(ptrs,m,d['config']['local_start'],stages,grid,torch.cuda.current_stream(dev).cuda_stream,flags)
    return out


def routed_var(pr,rs,state,flags,stamps=None):
    x,ids,tw,lids=route_call(pr,rs)
    d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
    return var_call(d,state,flags,stamps=stamps),x,ids,tw,lids


def run_lengths(d,grid=264):
    """W2 tile runs per (CTA, expert, block) under the incumbent's unit schedule (host replica)."""
    ids=d['topk_ids'].flatten().tolist();n=[0]*32
    for e in ids:
        if 0<=e<32:n[e]+=1
    nblk=[(c+7)//8 for c in n];us=[0]
    for b in nblk:us.append(us[-1]+256*b)
    U=us[-1];cc=min(256,max(1,(U+grid//2)//grid));runs=[]
    for c in range(grid):
        a,b=U*c//grid,U*(c+1)//grid;cur=None;length=0
        for u in range(a,b):
            e=max(i for i in range(32) if us[i]<=u and nblk[i]);rel=u-us[e];span=cc*nblk[e];nfull=256//cc
            if rel<nfull*span:nb=(rel-(rel//span)*span)//cc
            else:nb=(rel-nfull*span)//(256-nfull*cc)
            if (e,nb)!=cur:
                if cur is not None:runs.append(length)
                cur=(e,nb);length=0
            length+=1
        if cur is not None:runs.append(length)
    return runs
