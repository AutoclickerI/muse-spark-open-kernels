"""Attribute the native-wide loss: original/lean TMA vs parallel cp.async X.

Five real whole-MoE arms on identical routes: S14 nf, original CS4, lean CS4,
cp.async CS1 and direct-row cp.async CS1. The last two keep native wide
accumulators and packed weights while replacing TMA activation staging.
DIRECT removes gather. Generic->async X publication retains its fence.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
import cluster_lean_build
from light_build import value,once

COPY=r'''
template<int NT,int J,int HALF>__device__ __forceinline__ void copyx(
 __nv_bfloat16*bs,const __nv_bfloat16*root,int st,int rem){
 int t=threadIdx.x,r=t/8+J*16,k=(t&7)*8;
 unsigned dst=mwbr::smaddr(bs+st*NT*128+HALF*NT*64+r*64+(k^((r&7)*8)));
 // One moving global root, with immediate row offsets, avoids NJ live u64
 // row pointers and per-row predicate registers in the wide token shapes.
 asm volatile("{ .reg .u64 p; .reg .u32 n; .reg .pred ok;"
              "setp.gt.s32 ok,%2,%4;selp.u32 n,16,0,ok;add.u64 p,%1,%3;"
              "cp.async.cg.shared.global [%0],[p],16,n;}"
              ::"r"(dst),"l"(root),"r"(rem),"n"(J*16*4096*2+HALF*64*2),"n"(J*16):"memory");
}
'''


def copy_cuda(cu):
    cu=once(cu,'namespace oc {','#include <utility>\nnamespace oc {\n'+COPY)
    a=cu.index(' auto issue=[&](int k,int st){');b=cu.index(' float acc[NT/2]',a)
    issue=cu[a:b]
    p=issue.index('#if BR_HOPPER');q=issue.index('#else',p);z=issue.index('#endif',q)
    issue=issue[:p]+'''  [&]<int...J>(std::integer_sequence<int,J...>){
   (copyx<NT,J,0>(bs,copy_root,st,count-row-t/8),...);
   (copyx<NT,J,1>(bs,copy_root,st,count-row-t/8),...);
  }(std::make_integer_sequence<int,NJ>{});
  copy_root+=128;
'''+issue[z+len('#endif'):]
    cu=cu[:a]+''' const __nv_bfloat16*copy_root=X+(size_t)(offset+row+t/8)*4096+(t&7)*8;
'''+issue+cu[b:]
    cu=once(cu,'#if BR_HOPPER\n  mbwait(smaddr(bar+st),(k/S)&1);\n#endif\n  __syncthreads();',
            '''  __syncthreads();
#if BR_HOPPER
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif''')
    # The activation buffers are local to each CTA in this arm. Remove only
    # the cluster/mbarrier lifecycle, keeping every WGMMA fence and wait.
    cu=once(cu,'#if BR_HOPPER\n cb();\n#endif\n}\n\ntemplate<int MAX', '}\n\ntemplate<int MAX')
    a=cu.index('#if BR_HOPPER\n if(threadIdx.x==0){',cu.index('template<int MAX,bool F,int CS>__global__'))
    b=cu.index('#endif',a)+len('#endif');cu=cu[:a]+cu[b:]
    a=cu.index('#if BR_HOPPER\n cb();\n if(threadIdx.x==0){',a)
    b=cu.index('#endif',a)+len('#endif');cu=cu[:a]+cu[b:]
    cu=once(cu,'CUtensorMap maps[2]={oc_map((void*)v[3],M*8+32*128,MAX-32),oc_map((void*)v[3],M*8+32*128,MAX)};', 'CUtensorMap maps[2]{};')
    return cu



def direct_cuda(cu):
    cu=once(cu,'namespace oc {','namespace oc {\n'+(H/'cluster_direct.cu.inc').read_text())
    cu=once(cu,'const int*C,const int*OFF,const __nv_bfloat16*PNW', 'const int*C,const int*OFF,const int*SLOTS,int M,const __nv_bfloat16*PNW')
    cu=once(cu,'(Q,SC,GS,X,C,OFF,PNW,ACT,Y,SS', '(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS')
    cu=once(cu,' const __nv_bfloat16*copy_root=X+(size_t)(offset+row+t/8)*4096+(t&7)*8;',
            """ int direct_src[NJ]={};
 if constexpr(F){
  [&]<int...J>(std::integer_sequence<int,J...>){
   ((direct_src[J]=(row+t/8+J*16<count)?SLOTS[e*M+row+t/8+J*16]/8:0),...);
  }(std::make_integer_sequence<int,NJ>{});
 }
 const __nv_bfloat16*copy_root=F?X+(t&7)*8:X+(size_t)(offset+row+t/8)*4096+(t&7)*8;""")
    cu=once(cu,'   (copyx<NT,J,0>(bs,copy_root,st,count-row-t/8),...);\n   (copyx<NT,J,1>(bs,copy_root,st,count-row-t/8),...);',
            """   if constexpr(F){
    (directx<NT,J,0>(bs,copy_root,st,count-row-t/8,direct_src[J]),...);
    (directx<NT,J,1>(bs,copy_root,st,count-row-t/8,direct_src[J]),...);
   }else{
    (copyx<NT,J,0>(bs,copy_root,st,count-row-t/8),...);
    (copyx<NT,J,1>(bs,copy_root,st,count-row-t/8),...);
   }""")
    a=cu.index('void oc_gather(');a=cu.index('{',a);b=cu.index('\n}',a)
    cu=cu[:a+1]+'(void)v;(void)gx;(void)M;(void)stream;'+cu[b:]
    return cu


def main():
    cluster_lean_build.main();s=(H/'cluster_payload.py').read_text();cs=value(s,'OP_CUDA');lean=cluster_lean_build.lean(cs[2]);copy=copy_cuda(lean);direct=direct_cuda(copy)
    tree=ast.parse(s);n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='OP_CUDA')
    lines=s.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['OP_CUDA='+repr([cs[1],cs[2],lean,copy,direct])+'\n'];s=''.join(lines)
    s=once(s,"OP_NAMES=('S12','N16G4','ONCE_CS1','ONCE_CS4')","OP_NAMES=('S14_NF','TMA_CS4','LEAN_CS4','COPY_CS1','DIRECT_CS1')")
    s=s.replace('if ai>=2','if ai>=1')
    s=once(s,'bases=(base,perm,perm,perm)','bases=(perm,perm,perm,perm,perm)')
    s=once(s,'(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()) if a==0 else (torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) if a==1 else',
             '(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) if a==0 else')
    s=once(s,'if arm==1:return nf_light_plan','if arm==0:return nf_light_plan')
    s=once(s,'if arm>=2:return oc_plan','if arm>=1:return oc_plan')
    s=once(s,'4 if arm==3 and torch.cuda.get_device_capability()==(9,0) else 1','4 if arm in (1,2) and torch.cuda.get_device_capability()==(9,0) else 1')
    s=s.replace("META S12 vs N16G4 vs heavy native N32..256 one-read/decode per expert tile, timed gather and compact S2/S1 pipeline. CS1 vs CS4 TMA multicast across adjacent channel tiles.",
                "META S14 nf f1c77ce6 vs TMA CS4 / lean TMA CS4 / parallel cp.async CS1. Same native N32..256, packed weights, timed gather and class streams; COPY removes TMA/multicast activation protocol and adds required generic-to-async publication fence.")
    s=s.replace('for a in range(4)','for a in range(5)')
    s=once(s,'def oc_plan(data,base,native,streams,cs):','def oc_plan(data,base,native,streams,cs,direct=False):')
    s=once(s,'gx=torch.empty_like(act);h13[3]=gx','gx=x if direct else torch.empty_like(act);h13[3]=gx')
    s=once(s,"else 1)\n    return nf_plan","else 1,arm==4)\n    return nf_plan")
    s=once(s,"    lines.append('OC_RES '","    lines.append('COPY_RES '+str([(n,f,list(OP_MODS[3].oc_info(n,f,1))) for n in (64,128,192,256) for f in (0,1)]))\n    lines.append('OC_RES '")
    s=once(s,'started=time.monotonic();op_build();rows=[];aa=[];peak=0.', 'started=time.monotonic();op_build();rows=[];aa=[];peak=0.;profiles=[]')
    s=once(s,'        bank=op_prepare(w)\n', '        bank=op_prepare(w)\n        if seed==0:profiles=oc_profiles(w,bank)\n')
    s=once(s,"    lines.append('COPY_RES '","    lines+=profiles\n    lines.append('COPY_RES '")
    profile=(H/'cluster_profile.py.inc').read_text().replace('for arm in range(4):','for arm in range(5):').replace('times=[[],[],[],[]];ratios=[[],[],[],[]]','times=[[] for _ in range(5)];ratios=[[] for _ in range(5)]').replace('range(1,4)','range(1,5)').replace('if arm==3 or','if arm>=3 or')
    profile=profile.replace('gather.append(capture(lambda v=v,gx=gx,mod=mod:mod.oc_gather(v,gx.data_ptr(),m,torch.cuda.current_stream().cuda_stream)))', 'gather.append(None if arm==4 else capture(lambda v=v,gx=gx,mod=mod:mod.oc_gather(v,gx.data_ptr(),m,torch.cuda.current_stream().cuda_stream)))')
    s+='\n'+profile
    compile(s,'<copy protocol lab>','exec');(H/'cluster_copy_payload.py').write_text(s);(H/'cu_cluster_copy.cu').write_text(copy);(H/'cu_cluster_direct.cu').write_text(direct)
    meta=dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),copy_cuda_sha256=hashlib.sha256(copy.encode()).hexdigest(),lean_cuda_sha256=hashlib.sha256(lean.encode()).hexdigest(),direct_cuda_sha256=hashlib.sha256(direct.encode()).hexdigest())
    (H/'cluster_copy_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)

if __name__=='__main__':main()
