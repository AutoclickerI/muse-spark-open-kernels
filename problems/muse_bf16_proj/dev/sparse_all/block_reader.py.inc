@triton.jit
def dc_block_context(bank):
    h=bank.to(tl.pointer_type(tl.uint32))
    return (bank+tl.load(h+6),(bank+tl.load(h+5)).to(tl.pointer_type(tl.uint32)),
        (bank+tl.load(h+7)).to(tl.pointer_type(tl.uint32)),tl.load(h+4),
        (bank+tl.load(h+8)).to(tl.pointer_type(tl.int32)),
        (bank+tl.load(h+9)).to(tl.pointer_type(tl.uint32)),
        (bank+tl.load(h+10)).to(tl.pointer_type(tl.uint32)),
        (bank+tl.load(h+11)).to(tl.pointer_type(tl.uint16)))


@triton.jit
def dc_block_load(ctx,row,col,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    # Decode each native1024-weight unit once. Broadcast its three compact
    # codewords and lane masks, then invert the native MMA-fragment permutation.
    NR:tl.constexpr=BN//16;NK:tl.constexpr=BK//64;U:tl.constexpr=NR*NK
    u=tl.arange(0,U);lane=tl.arange(0,32);j=tl.arange(0,32)
    rg=row//16+u//NK;unit=rg*(K//64)+col//64+u%NK
    block=ctx[0]+unit*1408
    cp=(block+1024).to(tl.pointer_type(tl.uint32))
    a=tl.load(cp[:,None]+lane[None,:]);b=tl.load(cp[:,None]+32+lane[None,:]);c=tl.load(cp[:,None]+64+lane[None,:])
    shift=j*3;q=shift//32;at=shift%32
    first=tl.where(q[None,None,:]==0,a[:,:,None],tl.where(q[None,None,:]==1,b[:,:,None],c[:,:,None]))
    second=tl.where(q[None,None,:]==0,b[:,:,None],tl.where(q[None,None,:]==1,c[:,:,None],0))
    symbol=(first>>at[None,None,:])|tl.where(at[None,None,:]>29,second<<((32-at[None,None,:])&31),0)
    low=tl.load(block[:,None,None]+(j//16)[None,None,:]*512+lane[None,:,None]*16+(j%16)[None,None,:]).to(tl.uint32)
    bit=tl.full((32,),1,tl.uint32)<<lane
    present=tl.load(ctx[1]+unit*2);off=tl.load(ctx[1]+unit*2+1)
    rank=tl.extra.cuda.libdevice.popc((present[:,None]&(bit[None,:]-1)).to(tl.int32))
    ext=tl.load(ctx[2]+off[:,None]+rank,(present[:,None]&bit[None,:])!=0,0)
    idx=(((symbol&3)|(((ext[:,:,None]>>j[None,None,:])&1)<<2))+ctx[3])&7
    base=tl.load(ctx[4]+rg).to(tl.uint32)
    value=(((symbol>>2)&1)<<15)|((base[:,None,None]+idx*2+(low>>7))<<7)|(low&127)
    ep=tl.load(ctx[5]+unit*2);eo=tl.load(ctx[5]+unit*2+1)
    has=(ep[:,None]&bit[None,:])!=0;er=tl.extra.cuda.libdevice.popc((ep[:,None]&(bit[None,:]-1)).to(tl.int32))
    em=tl.load(ctx[6]+(eo[:,None]+er)*2,has,0);vo=tl.load(ctx[6]+(eo[:,None]+er)*2+1,has,0)
    wbit=tl.full((32,),1,tl.uint32)<<j;escaped=has[:,:,None]&((em[:,:,None]&wbit[None,None,:])!=0)
    vr=tl.extra.cuda.libdevice.popc((em[:,:,None]&(wbit[None,None,:]-1)).to(tl.int32))
    exact=tl.load(ctx[7]+vo[:,:,None]+vr,escaped,0).to(tl.uint32)
    value=tl.where(escaped,exact,value).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    return value.reshape(NR,NK,8,4,2,2,2,2,2).permute(0,7,2,1,3,4,5,6,8).reshape(BN,BK)


@triton.jit
def dc_block_restore(BANK,OUT,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BN;col=tl.program_id(1)*BK;ctx=dc_block_context(BANK)
    tile=dc_block_load(ctx,row,col,K,BN,BK)
    r=row+tl.arange(0,BN);k=col+tl.arange(0,BK)
    tl.store(OUT+r[:,None]*K+k[None,:],tile)
