"""Native vq8 softmax: balanced row-sum tree and skip exp2(0) when the running max is unchanged."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent

def source():
    src=(HERE/'out/vq8_c262e4c5.cu').read_text()
    src=src.replace('    alpha[rh] = ex2(m[rh] - safe);', '    alpha[rh] = m[rh] == safe ? 1.f : ex2(m[rh] - safe);')
    src=src.replace('    float sum0 = 0.f, sum1 = 0.f;\n','')
    src=src.replace('      sum0 += p0;\n      sum1 += p1;\n','')
    old='    l[rh] = l[rh] * alpha[rh] + (sum0 + sum1);'
    assert old in src
    new='''    float ps[8];
#pragma unroll
    for (int n=0; n<8; n++) ps[n]=s[n][2*rh]+s[n][2*rh+1];
#pragma unroll
    for (int n=0; n<4; n++) ps[n]+=ps[n+4];
    const float total=(ps[0]+ps[2])+(ps[1]+ps[3]);
    l[rh] = fmaf(l[rh], alpha[rh], total);'''
    return src.replace(old,new)

if __name__=='__main__':
    s=source();p=HERE/'out/vq8_softmax.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
