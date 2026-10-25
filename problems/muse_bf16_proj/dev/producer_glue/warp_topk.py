"""Top8 of 256 FP32 ordered keys for a four-warp Triton CTA.

Two register values per thread -> each warp's top8 -> warp0 top8 of 32
candidates. Keys use integer redux; equal keys choose the lower expert id.
The helper returns 256 slots with only positions 0..7 live, ready for masked
stores. No global scratch or kernel launch is needed.
"""
import triton
import triton.language as tl


def asm_source():
    s=['{', '.reg .u32 tid,warp,lane,k0,k1,i0,i1,s0,s1,top,c0,c1,pick,chosen,q0,q1,base,off,addr,key,id,sc;',
       '.reg .f32 total,val,den;', '.reg .pred p,p0,p1,pwarp;',
       '.shared .align 4 .u32 gp_top8[113];',
       'mov.u32 tid,%tid.x; shr.u32 warp,tid,5; and.b32 lane,tid,31;',
       'mov.u32 k0,$4; mov.u32 k1,$5; mov.b32 s0,$6; mov.b32 s1,$7; mov.u32 i0,$8; mov.u32 i1,$9;',
       'mov.u32 base,gp_top8; shl.b32 off,warp,5; add.u32 base,base,off;',
       'setp.eq.u32 p,lane,0;']
    for r in range(8):
        s += ['max.u32 top,k0,k1; redux.sync.max.u32 top,top,0xffffffff;',
            'setp.eq.u32 p0,k0,top; selp.u32 c0,i0,0xffffffff,p0;',
            'setp.eq.u32 p1,k1,top; selp.u32 c1,i1,0xffffffff,p1;',
            'min.u32 pick,c0,c1; redux.sync.min.u32 pick,pick,0xffffffff;',
            'setp.eq.u32 p0,i0,pick; selp.u32 q0,s0,0,p0; @p0 mov.u32 k0,0;',
            'setp.eq.u32 p1,i1,pick; selp.u32 q1,s1,0,p1; @p1 mov.u32 k1,0;',
            'or.b32 chosen,q0,q1; redux.sync.or.b32 chosen,chosen,0xffffffff;',
            f'@p st.shared.u32 [base+{r*4}],top; @p st.shared.u32 [base+{128+r*4}],pick; @p st.shared.u32 [base+{256+r*4}],chosen;']
    s += ['bar.sync 0;', 'setp.eq.u32 pwarp,warp,0; @!pwarp bra GP_TOP_DONE;',
          'mov.u32 base,gp_top8; shl.b32 off,lane,2; add.u32 addr,base,off;',
          'ld.shared.u32 key,[addr]; ld.shared.u32 id,[addr+128]; ld.shared.u32 sc,[addr+256];',
          'mov.f32 total,0f00000000;']
    for r in range(8):
        s += ['redux.sync.max.u32 top,key,0xffffffff;',
            'setp.eq.u32 p0,key,top; selp.u32 pick,id,0xffffffff,p0;',
            'redux.sync.min.u32 pick,pick,0xffffffff;',
            'setp.eq.u32 p0,id,pick; selp.u32 chosen,sc,0,p0; @p0 mov.u32 key,0;',
            'redux.sync.or.b32 chosen,chosen,0xffffffff; mov.b32 val,chosen; add.f32 total,total,val;',
            f'@p st.shared.u32 [base+{384+r*4}],pick; @p st.shared.u32 [base+{416+r*4}],chosen;']
    s += ['add.f32 den,total,0f26901d7d; @p st.shared.f32 [base+448],den;',
          'GP_TOP_DONE: bar.sync 0;', 'mov.u32 base,gp_top8; ld.shared.f32 den,[base+448];',
          'mov.u32 $0,0; mov.u32 $1,0; mov.f32 $2,0f00000000; mov.f32 $3,0f00000000;',
          'setp.lt.u32 p0,i0,8; setp.lt.u32 p1,i1,8;',
          'shl.b32 off,i0,2; add.u32 addr,base,off;',
          '@p0 ld.shared.u32 $0,[addr+384]; @p0 ld.shared.f32 val,[addr+416]; @p0 div.rn.f32 $2,val,den;',
          'shl.b32 off,i1,2; add.u32 addr,base,off;',
          '@p1 ld.shared.u32 $1,[addr+384]; @p1 ld.shared.f32 val,[addr+416]; @p1 div.rn.f32 $3,val,den;', '}']
    return '\n'.join(s)


TOP8_TEXT=asm_source()
TOP8_ASM=tl.constexpr(TOP8_TEXT)


@triton.jit
def warp_top8(keys,scores,eids):
    return tl.inline_asm_elementwise(TOP8_ASM,'=r,=r,=f,=f,r,r,f,f,r,r',
        [keys,scores,eids],dtype=(tl.int32,tl.float32),is_pure=False,pack=2)


@triton.jit
def top8_kernel(SCORES,BIAS,IDS,WTS):
    e=tl.arange(0,256);scores=tl.load(SCORES+e);biased=scores+tl.load(BIAS+e)
    bits=biased.to(tl.int32,bitcast=True)
    ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
    ids,wts=warp_top8(ordered,scores,e)
    tl.store(IDS+e,ids,e<8);tl.store(WTS+e,wts,e<8)
