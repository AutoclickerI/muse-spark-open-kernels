"""Unroll the two-stage ring and use the explicit release/acquire cluster barrier.

The accepted original CUDA is preserved as an arm. Lean unrolls address/control
code, spells the cluster barrier explicitly, aliases norm scratch after wait0,
and assigns gather rows to warps. Weight precision and width classes are unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
from light_build import once,value
import cluster_build


def lean(cu):
    assert hashlib.sha256(cu.encode()).hexdigest()=='075ce977480876698ce94283907faafc8afab49b798b7e1d4a5b42d801d08c8d'
    cu=once(cu,'namespace oc {','''namespace oc {
__device__ __forceinline__ void cb(){
#if BR_HOPPER
 asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;":::"memory");
#else
 __syncthreads();
#endif
}
''')
    cu=once(cu,' __shared__ float red[F?1:4*MAX];', ' extern __shared__ __align__(1024) unsigned char ocsm[];float*red=(float*)ocsm;')
    cu=once(cu,'int z=blockIdx.x;if(z>=OFF[32])return;', 'int z=blockIdx.x*8+threadIdx.x/32;if(z>=OFF[32])return;')
    cu=once(cu,'for(int k=threadIdx.x*8;k<4096;k+=128*8)', 'for(int k=(threadIdx.x&31)*8;k<4096;k+=32*8)')
    cu=once(cu,'oc::gather<<<M*8+32*32,128,0,', 'oc::gather<<<(M*8+32*32+7)/8,256,0,')
    cu=once(cu,' auto cluster=cooperative_groups::this_cluster();','')
    cu=cu.replace('cluster.sync();','cb();').replace('cooperative_groups::this_cluster().sync();','cb();')
    cu=once(cu,'auto issue=[&](int k){int st=k%S;', 'auto issue=[&](int k,int st){')
    cu=once(cu,' issue(0);\n for(int k=0;k<32;++k){int st=k%S;', ''' issue(0,0);
 for(int k0=0;k0<32;k0+=S){
  #pragma unroll
  for(int st=0;st<S;++st){int k=k0+st;''')
    cu=cu.replace('issue(k+1);','issue(k+1,(st+1)%S);')
    cu=once(cu,'if constexpr(S==1)if(k+1<32)issue(k+1,(st+1)%S);\n }\n#if BR_HOPPER',
            'if constexpr(S==1)if(k+1<32)issue(k+1,(st+1)%S);\n  }\n }\n#if BR_HOPPER')
    return cu


def main():
    cluster_build.main();s=(H/'cluster_payload.py').read_text();cs=value(s,'OP_CUDA');new=lean(cs[2])
    n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='OP_CUDA')
    lines=s.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['OP_CUDA='+repr(cs+[new])+'\n'];s=''.join(lines)
    s=once(s,"OP_NAMES=('S12','N16G4','ONCE_CS1','ONCE_CS4')","OP_NAMES=('S12','N16G4','ONCE_CS1','ONCE_CS4','LEAN_CS4')")
    s=once(s,'bases=(base,perm,perm,perm)','bases=(base,perm,perm,perm,perm)')
    s=s.replace('for a in range(4)','for a in range(5)')
    s=once(s,'4 if arm==3 and torch.cuda.get_device_capability()==(9,0) else 1', '4 if arm>=3 and torch.cuda.get_device_capability()==(9,0) else 1')
    s=once(s,"    lines.append('OC_RES '","    lines.append('OC_LEAN_RES '+str([(n,f,list(OP_MODS[4].oc_info(n,f,4))) for n in (64,128,192,256) for f in (0,1)]))\n    lines.append('OC_RES '")
    compile(s,'<lean cluster>','exec');(H/'cluster_lean_payload.py').write_text(s);(H/'cu_cluster_lean.cu').write_text(new)
    manifest=dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(new.encode()).hexdigest(),base_cuda_sha256=hashlib.sha256(cs[2].encode()).hexdigest())
    (H/'cluster_lean_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)

if __name__=='__main__':main()
