"""Masked decode softmax plus CUDA's FP32 sincos intrinsic already used by the verify kernel."""
import build_masked,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def source():
    s=build_masked.source()
    old='if(w==0)csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);';assert old in s
    s=s.replace(old,'if(w==0){float sn,cs;__sincosf((float)p.positions[b]*p.inv_freq[lane],&sn,&cs);csbuf[lane]=make_float2(cs,sn);}')
    return s
if __name__=='__main__':
    s=source();p=HERE/'out/fast_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
