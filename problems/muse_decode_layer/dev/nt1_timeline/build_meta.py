"""Restore the MoERouteMeta consumer directly in frozen S11's NT1 kernel (flags101 vs flags69).
The producer is P7's existing cluster route kernel; this cell does not alter its geometry or serial epilogue.
"""
import ast,hashlib,re
from pathlib import Path
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
META_BRANCH='''  } else if constexpr (META) {
    if(tid<32){rt->n[tid]=rmeta[4+tid];rt->off[tid]=rmeta[36+tid];rt->nblk[tid]=rmeta[69+tid];}
    if(tid<33){rt->us1[tid]=rmeta[101+tid];rt->us2[tid]=rmeta[134+tid];}
    for(int i=tid;i<npairs;i+=NCT){pair_of[i]=static_cast<int16_t>(rmeta[199+i]);pos_of[i]=static_cast<int16_t>(rmeta[455+i]);}
    if(tid==0)bar_gen0=ld_acquire(cnt+C_GEN);
    __syncthreads();
    U1=rt->us1[NLOC];U2=rt->us2[NLOC];
    a1=static_cast<int>((long long)U1*c/G);b1=static_cast<int>((long long)U1*(c+1)/G);
    a2=static_cast<int>((long long)U2*c/G);b2=static_cast<int>((long long)U2*(c+1)/G);
    first_count=b1-a1;total_count=first_count+b2-a2;
    if(tid==0){
      rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));
      if(total_count){int e,nb,m;const int pass=first_count?0:1,cc=pass?rt->c2:rt->c1,ntile=pass?NT2:NT1;
        decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);
        rt->pre.pass=pass;rt->pre.e=e;rt->pre.nb=nb;rt->pre.m=m;rt->pre.mb=(m/cc)*cc;rt->pre.me=min(ntile,(m/cc)*cc+cc);}
      for(int i=0;i<nstage;++i)prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
    }
    __syncthreads();
  } else {
    if (tid < NLOC) rt->n[tid] = 0;'''

def build():
    s11=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text();assert hashlib.sha256(s11.encode()).hexdigest().startswith('acda6c62')
    tree=ast.parse(s11)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='SMALLM_VAR_CUDA' for t in n.targets))
    cuda=ast.literal_eval(node.value)
    assert cuda.count('int M, int local_start, int nstage, uint64_t* __restrict__ trace)')==1
    cuda=cuda.replace('int M, int local_start, int nstage, uint64_t* __restrict__ trace)', 'int M, int local_start, int nstage, uint64_t* __restrict__ trace, const int* __restrict__ rmeta)')
    cuda=cuda.replace('  F_Z = 1,', '  F_META = 32, // MoERouteMeta v1 from the completed cluster route producer\n  F_Z = 1,')
    cuda=cuda.replace('constexpr bool PDL = F & F_PDL,', 'constexpr bool META = F & F_META;\n  static_assert(!(META && SLOT), "META uses the general M>1 dispatch");\n  constexpr bool PDL = F & F_PDL,')
    anchor='  } else {\n    if (tid < NLOC) rt->n[tid] = 0;'
    assert cuda.count(anchor)==1;cuda=cuda.replace(anchor,META_BRANCH)
    anchor='(uint64_t*)(v.size() > 13 ? v[13] : 0)))'
    assert cuda.count(anchor)==1;cuda=cuda.replace(anchor,'(uint64_t*)(v.size() > 13 ? v[13] : 0), (const int*)(v.size() > 14 ? v[14] : 0)))')
    cuda=cuda.replace('#define VAR_FLAGS(X) X(69) X(95)', '#define VAR_FLAGS(X) X(69) X(101)')
    a=node.end_lineno;lines=s11.splitlines(True);start=s11.index('SMALLM_VAR_CPP = ',s11.index('SMALLM_VAR_CUDA = '));end=s11.index('def _s4_prepare_banks(',start)
    body='import hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nSMALLM_VAR_CUDA = '+repr(cuda)+'\n'+s11[start:end]
    body=body.replace('SMALLM_FLAGS_BUILT = [69, 95]', 'SMALLM_FLAGS_BUILT = [69, 101]')
    body=body.replace('ptrs = [v.data_ptr() for v in tensors]', "ptrs = [v.data_ptr() for v in tensors]+[0,data['route_meta'].data_ptr() if flags&32 else 0]")
    body=body.replace('smallm_', 'metam_').replace('SMALLM', 'METAM')
    (DEV/'meta_cell.py').write_text(body);(DEV/'meta.cu').write_text(cuda)
    print('meta_cell.py',hashlib.sha256(body.encode()).hexdigest())
if __name__=='__main__':build()
