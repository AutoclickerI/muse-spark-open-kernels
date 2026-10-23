"""Lossless V6 bank -> timed BF16 grouped-GEMM scratch, no raw bank needed."""
import torch
import triton
import triton.language as tl

@triton.jit
def _rotate_right(x, shift:tl.constexpr):
    if shift == 0:
        return x
    return (x >> shift) | (x << (32-shift))

@triton.jit
def _bf16x2_mul(a,b):
    return tl.inline_asm_elementwise('mul.rn.bf16x2 $0, $1, $2;', '=r,r,r',
                                    [a,b], dtype=tl.uint32, is_pure=True, pack=1)

@triton.jit
def _canonical_expand(WB,WW,OUT,COUNTS,ROWS:tl.constexpr,GATED:tl.constexpr,BLOCK:tl.constexpr):
    e=tl.program_id(0)
    if tl.load(COUNTS+e)==0:
        return
    tile=tl.program_id(1)
    # Each physical pair of words contains sixteen codes. Every packed byte is
    # read once; all eight decode slots have compile-time rotate/mask selectors.
    i=tl.arange(0,BLOCK)
    base=(e.to(tl.int64)*(ROWS//16)+tile)*36864
    A=tl.load(WW+base//4+i*2).to(tl.uint32)
    B=tl.load(WW+base//4+i*2+1).to(tl.uint32)
    W=(A & 0x60306030) | (B & 0x9fcf9fcf)
    lane=(i//2)%32
    kg=i//64
    p=i%2
    rr=lane//4
    q=lane%4
    for slot in tl.static_range(8):
        if slot==0:code=A
        elif slot==1:code=_rotate_right(A,4)
        elif slot==2:code=_rotate_right(A,10)
        elif slot==3:code=_rotate_right(B,2)
        elif slot==4:code=_rotate_right(B,6)
        elif slot==5:code=_rotate_right(B,12)
        elif slot==6:code=_rotate_right(W,24)
        else:code=_rotate_right(W,30)
        ri=slot%2
        sf=tl.load(WB+base+32768+(kg//8)*512+lane*16+(kg%8)*2+ri).to(tl.uint32)
        scale=0x70007000+sf*0x100010
        weight=_bf16x2_mul(_bf16x2_mul(code & 0x81c081c0,scale),0x43804380)
        if GATED:
            row=tile*8+rr+ri*(ROWS//2)
        else:
            row=tile*16+rr+ri*8
        pair=kg*32+q*8+p*4+(slot//4)*2+(slot//2)%2
        tl.store(OUT+(e.to(tl.int64)*ROWS+row)*2048+pair,weight)

def expand(bank,counts,rows,gated,*,warps=8,out=None):
    if out is None:
        out=torch.empty((bank.shape[0],rows,4096),dtype=torch.bfloat16,device=bank.device)
    _canonical_expand[(bank.shape[0],rows//16)](bank,bank.view(torch.int32),out.view(torch.int32),counts,
        ROWS=rows,GATED=gated,BLOCK=4096,num_warps=warps)
    return out

@triton.jit
def _canonical_expand_coalesced(WB,WW,OUT,COUNTS,ROWS:tl.constexpr,GATED:tl.constexpr,KB:tl.constexpr):
    e=tl.program_id(0)
    if tl.load(COUNTS+e)==0:return
    tile=tl.program_id(1)
    i=tl.arange(0,16*(KB//2))
    row16=i//(KB//2);kp=tl.program_id(2)*(KB//2)+i%(KB//2)
    ri=row16//8;rr=row16%8
    kg=kp//32;q=(kp//8)%4;p=(kp//4)%2
    lane=rr*4+q
    base=(e.to(tl.int64)*(ROWS//16)+tile)*36864
    srcword=base//4+kg*128+lane*4+p*2
    A=tl.load(WW+srcword).to(tl.uint32);B=tl.load(WW+srcword+1).to(tl.uint32)
    W=(A & 0x60306030) | (B & 0x9fcf9fcf)
    slot=(kp%4)*2+ri
    source=tl.where(slot<3,A,tl.where(slot<6,B,W))
    shifts=tl.where(ri==0,0xC1940,0xF3044).to(tl.uint32)
    rot=(shifts >> ((kp%4)*5)) & 31
    code=tl.inline_asm_elementwise('shf.r.wrap.b32 $0, $1, $1, $2;', '=r,r,r',
                                   [source,rot],dtype=tl.uint32,is_pure=True,pack=1)
    sf=tl.load(WB+base+32768+(kg//8)*512+lane*16+(kg%8)*2+ri).to(tl.uint32)
    scale=0x70007000+sf*0x100010
    weight=_bf16x2_mul(_bf16x2_mul(code & 0x81c081c0,scale),0x43804380)
    row=tile*8+rr+ri*(ROWS//2) if GATED else tile*16+row16
    tl.store(OUT+(e.to(tl.int64)*ROWS+row)*2048+kp,weight)
