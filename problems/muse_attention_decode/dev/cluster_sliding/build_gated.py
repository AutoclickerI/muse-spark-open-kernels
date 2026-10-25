"""Read the gate projection in the same cooperative pass; keep it in shared memory for the epilogue."""
import build_compact,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def source(late=False):
    s=build_compact.source()
    a=s.index('template<int QT>\nDI void prep_decode');b=s.index('template<int L,int QT,int NK>\nDI void prob(',a)
    p=s[a:b].replace('template<int QT>','template<int QT,bool GATE>').replace('R=(H+2+3)/4','R=(H+2+(GATE?H:0)+3)/4')
    p=p.replace('valid[j]=row<H+2;', 'valid[j]=row<H+2+(GATE?H:0);')
    old="row==H+1?(p.HQ+p.HK+kh)*64+2*lane:0;"
    assert old in p
    p=p.replace(old, "row==H+1?(p.HQ+p.HK+kh)*64+2*lane:(p.HQ+2*p.HK+kh*8+sub*H+row-H-2)*64+2*lane;\n    if(!valid[j])col[j]=0;")
    old='''    }
  }
  __syncthreads();proxy_fence();
}'''
    new='''    } else if constexpr(GATE) {
      if(row<2*H+2)*(float2*)(KN+1024+((row-H-2)*64+2*lane)*4)=y;
    }
  }
  __syncthreads();proxy_fence();
}'''
    assert old in p;p=p.replace(old,new)
    s=s[:a]+p+s[b:]
    s=s.replace('if constexpr(L==1)prep_decode<QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);',
        'if constexpr(L==1) {\n    if(p.gate)prep_decode<QT,true>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);\n    else prep_decode<QT,false>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);\n  }')
    old='if(p.gate) {int tk[1]={b*L+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 gv[1];input_pairs<1>(p,tk,col,valid,gv);gate=gv[0];}'
    new='if(p.gate) {\n      if constexpr(L==1)gate=*(float2*)(KN+1024+((r%H)*64+2*lane)*4);\n      else {int tk[1]={b*L+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 gv[1];input_pairs<1>(p,tk,col,valid,gv);gate=gv[0];}\n    }'
    assert old in s;s=s.replace(old,new)
    if late:
        old='  asm volatile("griddepcontrol.launch_dependents;":::"memory");\n';assert s.count(old)==1;s=s.replace(old,'')
        s=s.replace('  float O[8][4]={}',old+'  float O[8][4]={}')
    return s

if __name__=='__main__':
    s=source();p=HERE/'gated_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
