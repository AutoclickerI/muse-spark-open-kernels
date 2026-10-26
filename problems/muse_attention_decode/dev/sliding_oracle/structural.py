"""Remove per-key SFU exponentials with a bounded FP32 exponent approximation.

Every key still contributes. Online maxima, exact row rescaling, FP32 sums,
FP8 probability packing/PV and the original merge are retained.
"""
import hashlib,json,re
from pathlib import Path
import source as oracle
H=Path(__file__).resolve().parent
HELPER=r'''
// 2^x via an affine exponent/mantissa bit construction. The bias centers
// the normal-range relative error at about +/-3.04%; values remain FP32.
// Below the FP32 FTZ range return zero, as ex2.approx.ftz does.
DI float probability_exp(float x){
  if(x < -126.f)return 0.f;
  return __int_as_float(__float2int_rn(fmaf(x,8388608.f,1064992256.f)));
}
'''

def transform(s):
    anchor='template<int L,int QT,int NK>\nDI void prob('
    assert s.count(anchor)==1;s=s.replace(anchor,HELPER+'\n'+anchor)
    old='const float v=ex2(s[n][2*rh+e]-safe+8.f);';assert s.count(old)==1
    s=s.replace(old,'const float v=probability_exp(s[n][2*rh+e]-safe+8.f);')
    old='v0=ex2(v0-safe+8.f);v1=ex2(v1-safe+8.f);';assert s.count(old)==1
    s=s.replace(old,'v0=probability_exp(v0-safe+8.f);v1=probability_exp(v1-safe+8.f);')
    return s

def source(family,local=False):return transform(oracle.original(family,local))

def context_clock(family,candidate=False,local=False):
    s=oracle.source(family,local,True)
    if family=='v':
        # The first oracle receipt lacks these two verify stamps. Correct only
        # this new diagnostic, preserving the sent oracle builder and hashes.
        old='issue(0,0);issue(1,1);';assert s.count(old)==1;s=s.replace(old,old+'oracle_trace(p,1);')
        if local:
            old='__threadfence();wg_bar();';assert s.count(old)==1;s=s.replace(old,old+'oracle_trace(p,7);')
        else:
            old='__syncthreads();cbar();';assert s.count(old)==1;s=s.replace(old,old+'oracle_trace(p,7);')
    return transform(s) if candidate else s

def main():
    (H/'out').mkdir(exist_ok=True);meta={}
    for f in ('d','v'):
        s=source(f);p=H/'out'/('exp_'+f+'.cu');p.write_text(s);meta[f]=hashlib.sha256(s.encode()).hexdigest()
    (H/'structural_manifest.json').write_text(json.dumps(dict(mechanism='Per-key FP32 exponent construction, original online softmax and FP8 PV',sources=meta,oracle_job='f365764142034130',status='unmeasured until TASK and H200 pair'),indent=2)+'\n');print(meta)

if __name__=='__main__':main()
