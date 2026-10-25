"""Same native direct-copy device code: cluster1 / ordinary launch / serial classes.

The launch-mode comparison shares one compiled image. Query cluster occupancy
separately from per-SM block occupancy; those are different scheduler limits.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
import cluster_copy_build
from light_build import once,value

TOPO=r'''
template<int N,bool F>std::vector<int64_t> oc_topo(){
 int dev,major,sms;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));
 auto info=oc_resource<N,F,1>();if(major!=9)return {sms,info[4],-1};
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(sms*4);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=info[3];
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=1;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;
 int clusters;BR_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters,oc::kernel<N,F,1>,&cfg));return {sms,info[4],clusters};
}
std::vector<int64_t> oc_topology(int64_t n,int64_t f){
#define OC_TOP(N) if(n==N)return f?oc_topo<N,true>():oc_topo<N,false>();
 OC_TOP(64) OC_TOP(128) OC_TOP(192) OC_TOP(256)
#undef OC_TOP
 BR_REQUIRE(false);return {};
}
'''


def normal_cuda(cu):
    cu=once(cu,'int M,cudaStream_t stream){\n constexpr int S=', 'int M,cudaStream_t stream,bool regular){\n constexpr int S=')
    cu=once(cu,'if(major==9){BR_CUDA_CHECK(cudaLaunchKernelEx', 'if(major==9&&!regular){BR_CUDA_CHECK(cudaLaunchKernelEx')
    cu=once(cu,'if(maxnt==N&&cs==C){if(first)oc_launch', 'if(maxnt==N&&(cs==C||(cs==0&&C==1))){if(first)oc_launch')
    cu=cu.replace('oc_launch<N,true,C>(v,grid,M,(cudaStream_t)stream);', 'oc_launch<N,true,C>(v,grid,M,(cudaStream_t)stream,cs==0);')
    cu=cu.replace('oc_launch<N,false,C>(v,grid,M,(cudaStream_t)stream);', 'oc_launch<N,false,C>(v,grid,M,(cudaStream_t)stream,cs==0);')
    return cu+'\n'+TOPO



def deep_cuda(cu):
    # MAX64's three slots fit three CTAs/SM at the observed register footprint.
    # Larger classes retain their old shared-memory geometry.
    cu=cu.replace('MAX<=128?2:1','MAX<=64?3:MAX<=128?2:1')
    a=cu.index('template<int NT,bool F,int CS,int S>');b=cu.index('template<int MAX,bool F,int CS>__global__',a)
    tile=cu[a:b]
    tile=once(tile,' issue(0,0);', ' #pragma unroll\n for(int p=0;p<(S>1?S-1:1);++p)issue(p,p);')
    tile=once(tile,'for(int st=0;st<S;++st){int k=k0+st;', 'for(int st=0;st<S;++st){int k=k0+st;if(k>=32)continue;')
    tile=once(tile,'asm volatile("cp.async.wait_group 0;":::"memory");', 'asm volatile("cp.async.wait_group %0;"::"n"(S>1?S-2:0):"memory");')
    tile=once(tile,'if constexpr(S==2)if(sub==0&&k+1<32)issue(k+1,(st+1)%S);',
              'if constexpr(S>1)if(sub==0){if(k+S-1<32)issue(k+S-1,(st+S-1)%S);else asm volatile("cp.async.commit_group;":::"memory");}')
    return cu[:a]+tile+cu[b:]


def main():
    cluster_copy_build.main();s=(H/'cluster_copy_payload.py').read_text();cs=value(s,'OP_CUDA');cu=normal_cuda(cs[4]);deep=deep_cuda(cu)
    host_boundary='template<int MAX,bool F,int CS> void oc_launch('
    assert cu.split(host_boundary)[0]==cs[4].split(host_boundary)[0]
    tree=ast.parse(s);lines=s.splitlines(keepends=True);repls=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='OP_CUDA':repls.append((n.lineno,n.end_lineno,'OP_CUDA='+repr([cs[0],cu,cu,cu,deep])+'\n'))
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='OP_CPP':repls.append((n.lineno,n.end_lineno,'OP_CPP='+repr(value(s,'OP_CPP')+'std::vector<int64_t> oc_topology(int64_t,int64_t);')+'\n'))
        if isinstance(n,ast.FunctionDef) and n.name=='oc_plan':
            f=ast.get_source_segment(s,n).replace('cs,direct=False):','cs,direct=False,serial=False):')
            a=f.index('        for side,join,maxnt in zip(');b=f.index('        sides[4].wait_event',a)
            original=f[a:b]
            f=f[:a]+'''        if serial:
            sides[0].wait_event(fork)
            with torch.cuda.stream(sides[0]):
                for maxnt in (64,128,192,256):
                    mod.oc_run(p13,maxnt,1,cs,grid,m,sides[0].cuda_stream);mod.oc_run(p2,maxnt,0,cs,grid,m,sides[0].cuda_stream)
                joins[0].record()
        else:
'''+''.join('    '+line+'\n' for line in original.splitlines())+f[b:]
            f=once(f,'for join in joins:parent.wait_event(join)', 'for join in ([joins[0],joins[4]] if serial else joins):parent.wait_event(join)')
            repls.append((n.lineno,n.end_lineno,f+'\n'))
        if isinstance(n,ast.FunctionDef) and n.name=='op_plan':
            f='''def op_plan(data,bank,arm):
    global OP_ARM
    OP_ARM=arm
    if arm==0:return nf_light_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm],4)
    return oc_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm],1 if arm==1 else 0,True,arm>=3)
'''
            repls.append((n.lineno,n.end_lineno,f))
        if isinstance(n,ast.FunctionDef) and n.name=='oc_profiles':
            f=(H/'cluster_profile.py.inc').read_text().replace('for arm in range(4):','for arm in range(5):').replace('times=[[],[],[],[]];ratios=[[],[],[],[]]','times=[[] for _ in range(5)];ratios=[[] for _ in range(5)]').replace('range(1,4)','range(1,5)')
            f=once(f,'gather.append(capture(lambda v=v,gx=gx,mod=mod:mod.oc_gather(v,gx.data_ptr(),m,torch.cuda.current_stream().cuda_stream)))','gather.append(None)')
            a=f.index('                        for side,join,n in zip(');b=f.index('                phases.append(capture(heavy))',a)
            f=f[:a]+'''                        if arm>=3:
                            sides[0].wait_event(fork)
                            with torch.cuda.stream(sides[0]):
                                for n in (64,128,192,256):mod.oc_run(p,n,first,0,sms*4,m,sides[0].cuda_stream)
                                joins[0].record()
                            parent.wait_event(joins[0])
                        else:
                            for side,join,n in zip(sides,joins,(64,128,192,256)):
                                side.wait_event(fork)
                                with torch.cuda.stream(side):mod.oc_run(p,n,first,1 if arm==1 else 0,sms*4,m,side.cuda_stream);join.record()
                            for join in joins:parent.wait_event(join)
'''+f[b:]
            repls.append((n.lineno,n.end_lineno,f+'\n'))
    for a,b,f in sorted(repls,reverse=True):lines[a-1:b]=[f]
    s=''.join(lines)
    s=once(s,"OP_NAMES=('S14_NF','TMA_CS4','LEAN_CS4','COPY_CS1','DIRECT_CS1')","OP_NAMES=('S14_NF','DIRECT_CLUSTER1','DIRECT_NORMAL','DIRECT_SERIAL','DEEP_SERIAL')")
    s=once(s,"'oc_run','oc_info','oc_gather']","'oc_run','oc_info','oc_gather','oc_topology']")
    s=once(s,"    lines.append('COPY_RES '","    lines.append('TOPOLOGY '+str([(n,f,list(OP_MODS[1].oc_topology(n,f))) for n in (64,128,192,256) for f in (0,1)]))\n    lines.append('COPY_RES '")
    s=once(s,"    lines.append('COPY_RES '","    lines.append('DEEP_RES '+str([(n,f,list(OP_MODS[4].oc_info(n,f,1))) for n in (64,128,192,256) for f in (0,1)]))\n    lines.append('COPY_RES '")
    s=s.replace('all four arms compute','all five arms compute')
    s=s.replace('Same native N32..256, packed weights, timed gather and class streams;', 'Same native N32..256, packed weights and no-gather class streams; DEEP uses S3 for MAX64 and retains S2/S1 elsewhere;')
    s=s.replace('META S14 nf f1c77ce6 vs TMA CS4 / lean TMA CS4 / parallel cp.async CS1.', 'META S14 nf f1c77ce6 vs identical DIRECT device image: cluster1 launch / ordinary launch / ordinary launch with serial classes. No gather in all direct arms. Cluster capacity reported separately from per-SM block capacity.')
    compile(s,'<schedule lab>','exec');(H/'cluster_schedule_payload.py').write_text(s);(H/'cu_cluster_schedule.cu').write_text(cu);(H/'cu_cluster_deep.cu').write_text(deep)
    meta=dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),device_base_cuda_sha256=hashlib.sha256(cs[4].encode()).hexdigest(),deep_cuda_sha256=hashlib.sha256(deep.encode()).hexdigest(),device_prefix_unchanged=True,cluster_normal_serial_share_image=True)
    (H/'cluster_schedule_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)

if __name__=='__main__':main()
