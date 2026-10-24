"""Execute the observed sm90a FP8-mma lowering explicitly on the local sm120 GPU."""

def forced_lowering(source):
    start=source.index('DI void mma8(');end=source.index('\nstruct State',start)
    # One FP8 PTX operation becomes twelve packed conversions and two FP16
    # m16n8k16 operations in the frozen CUDA13 sm90a probe's SASS.
    cvt=[]
    for i,slot in enumerate(range(4,10)):
        prefix='a' if i<4 else 'b';j=i if i<4 else i-4
        cvt.extend([f'mov.b32 {{lo,hi}},%{slot};',f'cvt.rn.f16x2.e4m3x2 {prefix}l{j},lo;',f'cvt.rn.f16x2.e4m3x2 {prefix}h{j},hi;'])
    body=' '.join(cvt)
    for tag in ('l','h'):
        body+=f' mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%0,%1,%2,%3}},{{a{tag}0,a{tag}1,a{tag}2,a{tag}3}},{{b{tag}0,b{tag}1}},{{%0,%1,%2,%3}};'
    helper='''DI void mma8(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("{.reg .b16 lo,hi;.reg .b32 al<4>,ah<4>,bl<2>,bh<2>; BODY }"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}'''.replace('BODY',body)
    return source[:start]+helper+source[end:]
