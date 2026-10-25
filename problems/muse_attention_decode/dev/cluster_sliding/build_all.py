"""One nonspinning last-CTA launch for both decode and8-token sliding verify."""
import hashlib,build_fast
from pathlib import Path
HERE=Path(__file__).resolve().parent

def source():
    s=build_fast.source()
    s=s.replace('if(L!=1 || ns<1','if((L!=1 && L!=8) || ns<1')
    s=s.replace('else throw std::runtime_error("decode only");','else {if(qt==4){RES(8,4);}else{RES(8,1);}}')
    old='  if(L==1){if(qt==4){GO(1,4);}else{GO(1,1);}}';assert old in s
    s=s.replace(old,old+'\n  else {if(qt==4){GO(8,4);}else{GO(8,1);}}')
    old='const int head=kh*8+sub*H+r;float2 gate=make_float2(0.f,0.f);';assert old in s
    s=s.replace(old,'const int token=r/H,head=kh*8+sub*H+r%H;float2 gate=make_float2(0.f,0.f);')
    old='if(p.gate)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);';assert old in s
    s=s.replace(old,'''if(p.gate) {
      if constexpr(L==1)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);
      else {int tk[1]={b*L+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 v[1];input_pairs<1>(p,tk,col,valid,v);gate=v[0];}
    }''')
    s=s.replace('((int64_t)b*p.HQ+head)*64+2*lane','((int64_t)(b*L+token)*p.HQ+head)*64+2*lane')
    return s

if __name__=='__main__':
    s=source();p=HERE/'out/all_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
