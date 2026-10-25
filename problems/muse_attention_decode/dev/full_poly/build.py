"""Replace probability exponentials with a short FP32 range-reduced polynomial; retain FP32 max rescaling."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent

def source(degree=3):
    s=(HERE.parent/'vq8/out/vq8_c262e4c5.cu').read_text()
    assert hashlib.sha256(s.encode()).hexdigest().startswith('c262e4c5')
    pol={2:'fmaf(fmaf(0.24229447f,f,0.70150876f),f,0.99994824f)',
         3:'fmaf(fmaf(fmaf(0.05587551f,f,0.24229447f),f,0.69312727f),f,0.99994824f)'}[degree]
    helper='''// Probabilities: x<=8, masked -inf becomes negligible FP32 mass (and exact FP8 zero).
// Relative error over a fractional exponent [-.5,.5]: degree2 .003763, degree3 .000189.
// Range reduction uses integer rounding; the polynomial and accumulation remain FP32.
DI float prob_ex2(float x) {
  x=fmaxf(x,-126.f);
  const int n=__float2int_rn(x);const float f=x-(float)n;
  const float y=POLY;
  return __uint_as_float(__float_as_uint(y)+(uint32_t)n*0x800000u);
}
'''.replace('POLY',pol)
    s=s.replace('// Online softmax of one 64-key tile;',helper+'\n// Online softmax of one 64-key tile;')
    a=s.index('DI void softmax(');b=s.index('DI void rescale(',a)
    sm=s[a:b].replace('ex2(fmaf(', 'prob_ex2(fmaf(')
    assert sm.count('prob_ex2(')==2
    return s[:a]+sm+s[b:]

if __name__=='__main__':
    for d in (2,3):
        s=source(d);p=HERE/f'poly{d}.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
