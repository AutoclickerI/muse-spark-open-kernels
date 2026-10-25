"""Native wide RS-WGMMA decode-once + TMA activation multicast, N16G4 light path kept."""
import ast
import hashlib
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H),str(H.parent),str(R/'tools')]
import build,patch_s12_permw,light_build
from build_prefill_lab import subset
from light_build import value,once

CPP='void oc_gather(std::vector<int64_t>,int64_t,int64_t,int64_t);std::vector<int64_t> oc_info(int64_t,int64_t,int64_t);void oc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'

def mma():
    body='template<int NT>__device__ __forceinline__ void mm(float*c,const unsigned*a,const __nv_bfloat16*xs,int b,uint64_t desc){\n#if BR_HOPPER\n uint64_t d=desc+b*2;\n'
    for nt in range(32,257,32):
        out=','.join('%'+str(i) for i in range(nt//2));aa=','.join('%'+str(nt//2+i) for i in range(4))
        operands=','.join('"+f"(c[%d])'%i for i in range(nt//2))
        body+=(' if constexpr' if nt==32 else ' else if constexpr')+f'(NT=={nt}){{\n'
        body+=f' asm volatile("{{ .reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n{nt}k16.f32.bf16.bf16 {{{out}}},{{{aa}}},%{nt//2+4},p,1,1,0;}}"\n :{operands}:"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }}\n'
    body+='\n#else\n mwbr::mma<32>(c,a,xs,b,desc);\n'
    # Portable MMA stand-in at every token octet, same ownership and layout.
    body=body.replace(' mwbr::mma<32>(c,a,xs,b,desc);', ''' int lane=threadIdx.x&31;
 #pragma unroll
 for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;
  unsigned b0=*(const unsigned*)(xs+r*64+(k^((r&7)*8)));
  unsigned b1=*(const unsigned*)(xs+r*64+((k+8)^((r&7)*8)));float*p=c+4*j;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
   :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
 }''')
    return body+'\n#endif\n}\n'

HOST=r'''
static CUtensorMap oc_map(void*ptr,uint64_t rows,int nt){
 static PFN_cuTensorMapEncodeTiled_v12000 enc=nullptr;
 if(!enc){cudaDriverEntryPointQueryResult q;BR_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&enc,cudaEnableDefault,&q));BR_REQUIRE(q==cudaDriverEntryPointSuccess&&enc);}
 CUtensorMap m;cuuint64_t dims[2]={4096,rows},strides[1]={8192};cuuint32_t box[2]={64,(unsigned)nt},es[2]={1,1};
 BR_REQUIRE(enc(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS);return m;
}
void oc_gather(std::vector<int64_t>v,int64_t gx,int64_t M,int64_t stream){
 oc::gather<<<M*8+32*32,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)v[3],(__nv_bfloat16*)gx,(int*)v[4],(int*)v[5],(int*)v[8],M);BR_CUDA_CHECK(cudaGetLastError());
}
// Each class uses a tensor-map box fixed at its MAX. The kernel slices the
// same MAX-row TMA stage into actual NT columns, preserving stride MAX below.
#define OC_CASES(F) F(64,1) F(128,1) F(192,1) F(256,1) F(64,4) F(128,4) F(192,4) F(256,4)
// Maps are a pair: NT=MAX-32 / MAX. Encoding is metadata-only, before launch.
template<int MAX,bool F,int CS> void oc_launch(std::vector<int64_t>&v,int grid,int M,cudaStream_t stream){
 constexpr int S=MAX<=128?2:1;
 CUtensorMap maps[2]={oc_map((void*)v[3],M*8+32*128,MAX-32),oc_map((void*)v[3],M*8+32*128,MAX)};
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=oc::dyn<S,MAX>();cfg.stream=stream;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=CS;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;
 int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));
 if(major==9){BR_CUDA_CHECK(cudaLaunchKernelEx(&cfg,oc::kernel<MAX,F,CS>,BR_ARGS(v),M,0x70007000u,maps[0],maps[1]));}
 else{oc::kernel<MAX,F,CS><<<grid,128,oc::dyn<S,MAX>(),stream>>>(BR_ARGS(v),M,0x70007000u,maps[0],maps[1]);BR_CUDA_CHECK(cudaGetLastError());}
}
void oc_run(std::vector<int64_t>v,int64_t maxnt,int64_t first,int64_t cs,int64_t grid,int64_t M,int64_t stream){
#define OC_GO(N,C) if(maxnt==N&&cs==C){if(first)oc_launch<N,true,C>(v,grid,M,(cudaStream_t)stream);else oc_launch<N,false,C>(v,grid,M,(cudaStream_t)stream);return;}
 OC_CASES(OC_GO)
#undef OC_GO
 BR_REQUIRE(false);
}
template<int MAX,bool F,int CS> std::vector<int64_t> oc_resource(){
 constexpr int S=MAX<=128?2:1;cudaFuncAttributes a{};int n;
 BR_CUDA_CHECK(cudaFuncSetAttribute(oc::kernel<MAX,F,CS>,cudaFuncAttributeMaxDynamicSharedMemorySize,oc::dyn<S,MAX>()));
 BR_CUDA_CHECK(cudaFuncGetAttributes(&a,oc::kernel<MAX,F,CS>));
 BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,oc::kernel<MAX,F,CS>,128,oc::dyn<S,MAX>()));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,oc::dyn<S,MAX>(),n};
}
std::vector<int64_t> oc_info(int64_t maxnt,int64_t first,int64_t cs){
#define OC_INFO(N,C) if(maxnt==N&&cs==C){return first?oc_resource<N,true,C>():oc_resource<N,false,C>();}
 OC_CASES(OC_INFO)
#undef OC_INFO
 BR_REQUIRE(false);return {};
}
'''

def cuda(c):
    extra=(H/'cluster.cu.inc').read_text().replace('\nOC_MMA\n','\n'+mma()+'\n')
    # Pass one map per actual NT; a MAX-row map would have the wrong K-half
    # stride for the lower NT in this class.
    extra=once(extra,'const __grid_constant__ CUtensorMap map){','const __grid_constant__ CUtensorMap maplo,const __grid_constant__ CUtensorMap maphi){')
    extra=once(extra,'e,row,mt,bar,&map,scale_base,red)', 'e,row,mt,bar,(NT==MAX?&maphi:&maplo),scale_base,red)')
    return c+'\n'+extra+'\n'+HOST


def main():
    text=(R/'problems/muse_decode_layer/cand_r3_best7_s12_final.py').read_text();perm=patch_s12_permw.patch(text)
    light_build.main();light=(H/'light_payload.py').read_text();lc=value(light,'OP_CUDA')[3]
    cu=cuda(lc);base=value(text,'NF_CUDA')
    funcs=build.strip_imports(subset(perm,('nf_plan','nf_call','nf_ilv','nf_wide','nf_perm')))
    lp=ast.get_source_segment(light,next(n for n in ast.parse(light).body if isinstance(n,ast.FunctionDef) and n.name=='nf_light_plan'))
    plan=ast.get_source_segment(perm,next(n for n in ast.parse(perm).body if isinstance(n,ast.FunctionDef) and n.name=='nf_plan'))
    plan=plan.replace('def nf_plan(data,base,native,streams):','def oc_plan(data,base,native,streams,cs):')
    plan=once(plan,'    p13,p2,pn=', '    gx=torch.empty_like(act);h13[3]=gx\n    l13=list(h13);l2=list(h2);l13[3]=x;l13[7]=tasks[2];l2[7]=tasks[2]\n    pl13=[t.data_ptr() for t in l13];pl2=[t.data_ptr() for t in l2]\n    p13,p2,pn=')
    plan=once(plan,'owners=(h13,h2,nv,ids,tw,pos,tasks)','owners=(h13,h2,nv,ids,tw,pos,tasks,gx,l13,l2)')
    plan=once(plan,'side,fork,join=streams;grid=1056 if m<=128 else 2112', 'sides,fork,joins=streams;grid=sms*4')
    a=plan.index('    def body():');b=plan.index('    def finish():',a)
    plan=plan[:a]+'''    def body():
        _=owners;parent=torch.cuda.current_stream()
        # Gather is part of the timed graph. Native NT1 still reads original X.
        pv=list(p13);pv[3]=x.data_ptr();mod.oc_gather(pv,gx.data_ptr(),m,parent.cuda_stream)
        fork.record(parent)
        for side,join,maxnt in zip(sides[:4],joins[:4],(64,128,192,256)):
            side.wait_event(fork)
            with torch.cuda.stream(side):
                mod.oc_run(p13,maxnt,1,cs,grid,m,side.cuda_stream);mod.oc_run(p2,maxnt,0,cs,grid,m,side.cuda_stream);join.record()
        sides[4].wait_event(fork)
        with torch.cuda.stream(sides[4]):
            mod.br_run(pl13,70,sms*4,m,sides[4].cuda_stream);mod.br_run(pl2,71,sms*4,m,sides[4].cuda_stream);joins[4].record()
        mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)
        for join in joins:parent.wait_event(join)
'''+plan[b:]
    body=build.helpers()+'\n'+build.IMPORTS+'\n'+build.strip_imports(build.HH_CELL.read_text())+'\n'+funcs+'\n'+lp+'\n'+plan+'\n'
    body+=f'OP_CUDA={[base,lc,cu,cu]!r}\nOP_CPP={value(text,"NF_CPP")+CPP!r}\nNF_OPTS={{}}\n'
    rt=(H/'order_runtime.py.inc').read_text()
    rt=once(rt,"OP_NAMES=('S12','PERMW','ORDER','PERMW_ORDER')","OP_NAMES=('S12','N16G4','ONCE_CS1','ONCE_CS4')")
    rt=once(rt,'for c in OP_CUDA:', 'for ai,c in enumerate(OP_CUDA):')
    rt=once(rt,'cpp_sources=OP_CPP,cuda_sources=c,',"cpp_sources=OP_CPP if ai>=2 else OP_CPP.split('void oc_gather(')[0],cuda_sources=c,")
    rt=once(rt,"'pmz_resident','pmz_run'],","'pmz_resident','pmz_run']+(['oc_run','oc_info','oc_gather'] if ai>=2 else []),")
    rt=once(rt,'mod.br_init();NF_OPTS', '\n        if ai>=2:[mod.oc_info(n,f,c) for n in (64,128,192,256) for f in (0,1) for c in (1,4)]\n        mod.br_init();NF_OPTS')
    rt=once(rt,"next(t.device for t in weights.values() if torch.is_tensor(t))",
            "next(t.device for t in weights.values() if torch.is_tensor(t)) if weights is not None else torch.device('cuda')")
    rt=once(rt,'return dict(bases=(base,perm,base,perm),native=native,\n                streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()) for _ in OP_NAMES])',
            'return dict(bases=(base,perm,perm,perm),native=native,\n                streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()) if a==0 else (torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) if a==1 else ([torch.cuda.Stream() for _ in range(5)],torch.cuda.Event(),[torch.cuda.Event() for _ in range(5)]) for a in range(4)])')
    rt=once(rt,"    return nf_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm])", "    if arm==1:return nf_light_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm],4)\n    if arm>=2:return oc_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm],4 if arm==3 and torch.cuda.get_device_capability()==(9,0) else 1)\n    return nf_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm])")
    rt=once(rt,"    for m in (128,256):\n        rr=","    lines.append('OC_RES '+str([(n,f,list(OP_MODS[2].oc_info(n,f,4))) for n in (64,128,192,256) for f in (0,1)]))\n    for m in (128,256):\n        rr=")
    rt=rt.replace('META exact S12 final 240248b2 vs PERMW, ORDER, PERMW_ORDER; same N32 tile math, geometry and global weight reads. ORDER changes only (expert,token-group,channel) to (expert,channel,token-group), shortening inter-CTA weight reuse distance.', 'META S12 vs N16G4 vs heavy native N32..256 one-read/decode per expert tile, timed gather and compact S2/S1 pipeline. CS1 vs CS4 TMA multicast across adjacent channel tiles. Light N16G4 retained: NT1<=8, N16 for9-16. Whole MoE reads/decodes each expert weight tile once up to256 tokens. New heavy count classes on four streams. Same precision/work.')
    body+=rt;compile(body,'<cluster>','exec');(H/'cluster_payload.py').write_text(body);(H/'cu_cluster.cu').write_text(cu)
    meta=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest());(H/'cluster_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)

if __name__=='__main__':main()
