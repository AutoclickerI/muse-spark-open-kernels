"""Independent producer edits on frozen S15; compose full and leave-one-out arms."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[1]
BASE=P/'cand_r3_best7_s15.py'
SHA='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
RELEASE='asm volatile("griddepcontrol.launch_dependents;" ::: "memory");'
WAIT='asm volatile("griddepcontrol.wait;" ::: "memory");'
LAUNCH='''template<class K,class...A> static cudaError_t pdl_launch(K k,dim3 grid,dim3 block,size_t smem,cudaStream_t stream,A...a){
 cudaLaunchAttribute at{};at.id=cudaLaunchAttributeProgrammaticStreamSerialization;at.val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t cfg{};cfg.gridDim=grid;cfg.blockDim=block;cfg.dynamicSmemBytes=smem;cfg.stream=stream;cfg.attrs=&at;cfg.numAttrs=1;
 return cudaLaunchKernelEx(&cfg,k,a...);
}
'''


def compose(without=()):
    src=BASE.read_text();assert hashlib.sha256(src.encode()).hexdigest()==SHA
    tree=ast.parse(src);replacements=[];changes=[]
    def one(s,a,b):
        assert s.count(a)==1,(a[:120],s.count(a));return s.replace(a,b)
    def literal(name,group,edit,producer):
        if group in without:return
        n=next(n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        old=ast.literal_eval(n.value);new=edit(old)
        replacements.append((n.lineno-1,n.end_lineno,name+' = '+repr(new)+'\n'))
        changes.append(dict(group=group,source=name,producer=producer,
            before=hashlib.sha256(old.encode()).hexdigest(),after=hashlib.sha256(new.encode()).hexdigest()))
    def attention(cu):
        cu=one(cu,'  __nv_bfloat162 qv[QR];', '  '+WAIT+'\n  __nv_bfloat162 qv[QR];')
        # Enable only the real launch, not the occupancy-query configuration.
        a=cu.index('cudaError_t launch_t(');b=cu.index('template <int L, int STAGES>\nint max_cluster_t',a)
        host=cu[a:b].replace('cudaLaunchAttribute at[1];','cudaLaunchAttribute at[2];')
        host=one(host,'  cfg.numAttrs = 1;', '  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n  at[1].val.programmaticStreamSerializationAllowed = 1;\n  cfg.numAttrs = 2;')
        return cu[:a]+host+cu[b:]
    literal('ATTN_PORT_CUDA_SRC','attention',attention,'attn_main consumer: wait before qkv; entry trigger already present')
    literal('SMALLM_VAR_CUDA','moe',lambda c:one(c,'  constexpr bool PDL = F & F_PDL, LAST = F & F_LAST, TRACE = F & F_TRACE;',
        '  constexpr bool PDL = F & F_PDL, LAST = F & F_LAST, TRACE = F & F_TRACE;\n  if constexpr(PDL) '+RELEASE), 'moe_var: implicit exit -> entry; existing input wait retained')
    literal('V9_CUDA_SRC','moe',lambda c:one(c,'    int M, int local_start, int nstage) {',
        '    int M, int local_start, int nstage) {\n  '+RELEASE), 'moe_fused: implicit exit -> entry; ordinary input launch retained')
    def nf(c):
        c=one(c,'namespace mwbr {',LAUNCH+'\nnamespace mwbr {')
        c=one(c,' int total=TASK[32]*MT;', ' '+WAIT+'\n int total=TASK[32]*MT;')
        c=one(c,' const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,unsigned scale_base){',
            ' const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,unsigned scale_base){\n '+RELEASE)
        c=one(c,' int t=threadIdx.x;\n __shared__ int count[32];', ' '+RELEASE+'\n int t=threadIdx.x;\n __shared__ int count[32];')
        c=one(c,' // All eight cold id loads per thread', ' '+WAIT+'\n // All eight cold id loads per thread')
        c=one(c,' constexpr int W=4096/SPLIT;', ' '+RELEASE+'\n '+WAIT+'\n constexpr int W=4096/SPLIT;')
        c=one(c,' __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){',
            ' __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\n '+RELEASE)
        c=one(c,' if(warp==0){int n=C[lane];', ' '+WAIT+'\n if(warp==0){int n=C[lane];')
        old='case ID:mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,mwbr::smem_bytes<NT>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,0x70007000u);break;'
        new='case ID:BR_CUDA_CHECK(pdl_launch(mwbr::gemm<NT,F,V,FM,LO,HI>,dim3(grid),dim3(128),mwbr::smem_bytes<NT>(),(cudaStream_t)stream,BR_ARGS(v),(int)M,0x70007000u));break;'
        c=one(c,old,new)
        # Transform the three remaining owned launch families, preserving all
        # arguments and event fork/join edges. Producers fire at entry; every
        # PDL consumer waits before its first predecessor-owned read.
        import re
        patterns=[(r'mwbr::route_live<(true|false)><<<1,256,0,\(cudaStream_t\)stream>>>\(([^;]+)\);',
                   r'BR_CUDA_CHECK(pdl_launch(mwbr::route_live<\1>,dim3(1),dim3(256),0,(cudaStream_t)stream,\2));',2),
                  (r'mwbr::finish<S><<<M\*S,256,0,\(cudaStream_t\)stream>>>\(([^;]+)\);',
                   r'BR_CUDA_CHECK(pdl_launch(mwbr::finish<S>,dim3(M*S),dim3(256),0,(cudaStream_t)stream,\1));',1),
                  (r'pmz::phase<1,F,true,Z,T><<<grid,pmz::NTHR,pmz::smem_bytes\(1,stages,M\),\(cudaStream_t\)stream>>>\(([^;]+)\);',
                   r'BR_CUDA_CHECK(pdl_launch(pmz::phase<1,F,true,Z,T>,dim3(grid),dim3(pmz::NTHR),pmz::smem_bytes(1,stages,M),(cudaStream_t)stream,\1));',1)]
        for pat,repl,count in patterns:
            c,n=re.subn(pat,repl,c);assert n==count,(pat,n)
        return c
    literal('NF_CUDA','moe',nf,'route_live,heavy/light gemm,phase,finish: entry releases and launch/wait pairs')
    lines=src.splitlines(True)
    for a,b,text in sorted(replacements,reverse=True):lines[a:b]=[text]
    src=''.join(lines)
    if 'attention' not in without:
        src=one(src,"name=f'muse_decode_attn_port_{arch}'", "name=f'muse_decode_attn_port_{hashlib.sha256(ATTN_PORT_CUDA_SRC.encode()).hexdigest()[:12]}_{arch}'")
    if 'projections' not in without:
        src=one(src,'    pid = tl.program_id(0)\n    ps = tl.program_id(1)\n    nw = N // BW',
            '    tl.extra.cuda.gdc_launch_dependents()\n    pid = tl.program_id(0)\n    ps = tl.program_id(1)\n    nw = N // BW')
        src=one(src,'num_warps=NWARPS, num_stages=NS)\n    return P, S',
            'num_warps=NWARPS, num_stages=NS, launch_pdl=True)\n    return P, S')
        # VDSK is first-stage qkv only; X is the caller-owned layer input, not
        # an output of an in-layer producer. No consumer wait is needed there.
        changes.append(dict(group='projections',source='_vdsk_gemm',producer='VDSK qkv: implicit exit -> entry; input independent'))
    return src,changes


def main():
    out=H/'out';out.mkdir(exist_ok=True)
    arms={}
    for name,without in [('E',()),('no_attention',('attention',)),('no_moe',('moe',)),
                         ('no_projections',('projections',)),('no_glue',('glue',))]:
        s,changes=compose(without)
        p=(P/'cand_r3_best7_s15e.py') if name=='E' else out/(name+'.py')
        p.write_text(s)
        arms[name]=dict(file=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),changes=changes)
    diff=list(difflib.unified_diff(BASE.read_text().splitlines(True),compose()[0].splitlines(True),n=0))
    meta=dict(base=str(BASE),base_sha256=SHA,arms=arms,glue='no-op: every selected owned glue kernel already releases at entry',
              diff=dict(add=sum(x.startswith('+') and not x.startswith('+++') for x in diff),
                        remove=sum(x.startswith('-') and not x.startswith('---') for x in diff)))
    # Count expanded CUDA changes separately from Python string serialization.
    strings=lambda text:{n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(text).body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
    old,new=strings(BASE.read_text()),strings(compose()[0]);expanded=[]
    for name in old:
        if old[name]==new.get(name):continue
        dd=list(difflib.unified_diff(old[name].splitlines(),new[name].splitlines(),n=0))
        expanded.append(dict(source=name,add=sum(l.startswith('+') and not l.startswith('+++') for l in dd),
            remove=sum(l.startswith('-') and not l.startswith('---') for l in dd)))
    meta['expanded_cuda_diff']=expanded
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('S15e',arms['E']['sha256'],meta['diff'])


if __name__=='__main__':main()
