"""Sparse high-index plane, exact point escapes, and native NF/CD stream adapters."""
import torch,hashlib,re
import triton
import triton.language as tl

# Index/sign positions in each native eight-weight codeword.
DC_POS={0:(tuple(range(0,32,4)),(3,19,7,23,11,27,15,31)),
        1:((0,8,16,24,4,12,20,28),(7,15,23,31,11,19,27,3)),
        2:((0,8,16,24,28,4,12,20),(7,15,23,31,3,11,19,27))}


def dc_pack(raw,fmt,*,base=None,escape=None,n=None,k=None):
    """Sparse compression of exact three-plane units; optional CD point-escape index."""
    dev=raw.device;u=raw.numel()//1536;a=raw.view(u,3,32,16)
    c=a[:,2].contiguous().view(torch.int32).long().reshape(u,32,4)&0xffffffff
    nib,sign=DC_POS[fmt];nib=torch.tensor(nib,device=dev);sign=torch.tensor(sign,device=dev)
    hi=((c[...,None]>>nib)&7).reshape(u,32,32);sg=((c[...,None]>>sign)&1).reshape(u,32,32)
    hist=torch.bincount(hi.flatten(),minlength=8)
    rot=int(torch.stack([hist[(torch.arange(4,device=dev)+i)%8].sum() for i in range(8)]).argmax())
    v=(hi-rot)&7;ext=v>>2;sym=(v&3)|(sg<<2);bits=torch.zeros((u,32,3),device=dev,dtype=torch.int64)
    for i in range(32):
        q,sh=divmod(i*3,32);bits[:,:,q]|=sym[:,:,i]<<sh
        if sh>29:bits[:,:,q+1]|=sym[:,:,i]>>(32-sh)
    small=bits.int().permute(0,2,1).contiguous().view(torch.uint8).reshape(u,384)
    masks=(ext<<torch.arange(32,device=dev)).sum(-1);used=masks!=0
    pres=(used.long()<<torch.arange(32,device=dev)).sum(-1);cnt=used.sum(-1);off=cnt.cumsum(0)-cnt
    emeta=torch.stack((pres,off),-1).int().contiguous().view(torch.uint8).flatten()
    data=torch.cat((a[:,:2].reshape(u,1024),small),1).flatten()
    sparse=masks[used].int().contiguous().view(torch.uint8).flatten()
    header=torch.zeros(32,device=dev,dtype=torch.int32);header[:5]=torch.tensor([0x5350414c,n or 0,k or 0,u,rot],device=dev,dtype=torch.int32)
    parts=[header.view(torch.uint8)];offset=128
    def add(index,value):
        nonlocal offset
        value=value.contiguous().view(torch.uint8).flatten();header[index]=offset
        parts.append(value);offset+=value.numel();pad=(-offset)%16
        if pad:parts.append(torch.zeros(pad,device=dev,dtype=torch.uint8));offset+=pad
    add(5,emeta);add(6,data);add(7,sparse);header[12]=fmt
    if base is not None:
        add(8,base.int())
        # Independent point access to the existing original-precision escapes.
        en=escape.long();rr=(en[:,0]>>16);cc=en[:,0]&65535
        # The caller attaches unit IDs (one per escaped entry) as a third column.
        unit=en[:,2];lane=(rr%8)*4+(cc%64)//16
        i=(cc//8%2)*16+(cc//4%2)*8+(cc//2%2)*4+(rr//8)*2+cc%2
        key=unit*32+lane;full=torch.zeros(u*32,device=dev,dtype=torch.int64)
        if en.numel():full.scatter_add_(0,key,1<<i)
        occupied=full!=0;presence=(occupied.view(u,32).long()<<torch.arange(32,device=dev)).sum(-1)
        counts=occupied.view(u,32).sum(-1);prefix=counts.cumsum(0)-counts
        group_mask=full[occupied];value_counts=torch.bincount(key,minlength=u*32)[occupied]
        voff=value_counts.cumsum(0)-value_counts
        order=(key*32+i).argsort();vals=en[order,1].to(torch.int16)
        add(9,torch.stack((presence,prefix),-1).int());add(10,torch.stack((group_mask,voff),-1).int());add(11,vals)
    packed=torch.cat(parts);packed[:128]=header.view(torch.uint8)
    # Inverse proof includes signs, rotated indices and every original low byte.
    rebuilt=torch.zeros_like(sym)
    for i in range(32):
        q,sh=divmod(i*3,32);v=bits[:,:,q]>>sh
        if sh>29:v|=bits[:,:,q+1]<<(32-sh)
        rebuilt[:,:,i]=v&7
    assert torch.equal((((rebuilt&3)|ext*4)+rot)&7,hi) and torch.equal(rebuilt>>2,sg)
    stats=dict(weights=u*1024,raw16_bytes=u*2048,raw12_bytes=raw.numel(),body_bytes=128+emeta.numel()+data.numel()+sparse.numel(),total_bytes=packed.numel(),bits_per_weight=packed.numel()*8/(u*1024),body_bits=(128+emeta.numel()+data.numel()+sparse.numel())*8/(u*1024),rotation=rot,occupied=float(used.float().mean()))
    return packed,stats


def dc_encode_cd(c,point=True):
    n,k=c['N'],c['K'];dev=c['coded'].device
    count=c['eoff'][1:]-c['eoff'][:-1]
    unit=torch.repeat_interleave(torch.arange(count.numel(),device=dev),count.long())
    esc=torch.cat((c['esc'][:c['nesc']],unit[:,None].int()),1)
    packed,stats=dc_pack(c['coded'],2,base=c['base'] if point else None,escape=esc if point else None,n=n,k=k)
    return dict(c,coded=packed),stats


@triton.jit
def dc_load(BANK,r,c,K:tl.constexpr):
    # One PTX expression keeps decoder temporaries out of Triton's software-pipeline
    # shared buffers; the original NS4 balanced schedules otherwise exceeded227KiB.
    bits=tl.inline_asm_elementwise(
'{\n.reg .u64 root, dense, meta, sparse, ptr, adr;\n.reg .u32 unit,lane,ix,r,c,kk,kb,rot,off,pres,start,bit,rank,ext,low,sh,q,at,a,b,v,idx,base,bits,mask,pos,val,tmp,tmp2;\n.reg .pred p;\nmov.u64 root,$1;mov.u32 r,$2;mov.u32 c,$3;mov.u32 kk,$4;\nshr.u32 kb,kk,6;shr.u32 unit,r,4;mul.lo.u32 unit,unit,kb;shr.u32 tmp,c,6;add.u32 unit,unit,tmp;\nand.b32 lane,r,7;shl.b32 lane,lane,2;and.b32 tmp,c,63;shr.u32 tmp,tmp,4;add.u32 lane,lane,tmp;\nshr.u32 ix,c,3;and.b32 ix,ix,1;shl.b32 ix,ix,4;\nshr.u32 tmp,c,2;and.b32 tmp,tmp,1;shl.b32 tmp,tmp,3;or.b32 ix,ix,tmp;\nshr.u32 tmp,c,1;and.b32 tmp,tmp,1;shl.b32 tmp,tmp,2;or.b32 ix,ix,tmp;\nshr.u32 tmp,r,3;and.b32 tmp,tmp,1;shl.b32 tmp,tmp,1;or.b32 ix,ix,tmp;and.b32 tmp,c,1;or.b32 ix,ix,tmp;\nld.global.nc.u32 rot,[root+16];ld.global.nc.u32 off,[root+20];cvt.u64.u32 ptr,off;add.u64 meta,root,ptr;\nld.global.nc.u32 off,[root+24];cvt.u64.u32 ptr,off;add.u64 dense,root,ptr;\nld.global.nc.u32 off,[root+28];cvt.u64.u32 ptr,off;add.u64 sparse,root,ptr;\nmul.wide.u32 adr,unit,8;add.u64 ptr,meta,adr;ld.global.nc.u32 pres,[ptr];ld.global.nc.u32 start,[ptr+4];\nshl.b32 bit,1,lane;sub.u32 tmp,bit,1;and.b32 tmp,tmp,pres;popc.b32 rank,tmp;and.b32 tmp,pres,bit;\nmov.u32 ext,0;setp.eq.u32 p,tmp,0;@p bra DC_EXT_DONE;\nadd.u32 tmp,start,rank;mul.wide.u32 adr,tmp,4;add.u64 ptr,sparse,adr;ld.global.nc.u32 ext,[ptr];\nDC_EXT_DONE:\nmul.wide.u32 adr,unit,1408;add.u64 dense,dense,adr;\nshr.u32 tmp,ix,4;shl.b32 tmp,tmp,9;shl.b32 tmp2,lane,4;add.u32 tmp,tmp,tmp2;and.b32 tmp2,ix,15;add.u32 tmp,tmp,tmp2;\ncvt.u64.u32 adr,tmp;add.u64 ptr,dense,adr;ld.global.nc.u8 low,[ptr];\nmul.lo.u32 sh,ix,3;shr.u32 q,sh,5;and.b32 at,sh,31;shl.b32 q,q,5;add.u32 q,q,lane;\nmul.wide.u32 adr,q,4;add.u64 ptr,dense,1024;add.u64 ptr,ptr,adr;ld.global.nc.u32 a,[ptr];mov.u32 b,0;\nsetp.le.u32 p,at,29;@p bra DC_CODE_DONE;ld.global.nc.u32 b,[ptr+128];\nDC_CODE_DONE:\nshr.u32 v,a,at;sub.u32 tmp,32,at;shl.b32 b,b,tmp;or.b32 v,v,b;\nand.b32 idx,v,3;shr.u32 tmp,ext,ix;and.b32 tmp,tmp,1;shl.b32 tmp,tmp,2;or.b32 idx,idx,tmp;add.u32 idx,idx,rot;and.b32 idx,idx,7;\nld.global.nc.u32 off,[root+32];cvt.u64.u32 ptr,off;add.u64 ptr,root,ptr;shr.u32 tmp,r,4;mul.wide.u32 adr,tmp,4;add.u64 ptr,ptr,adr;ld.global.nc.u32 base,[ptr];\nshl.b32 idx,idx,1;add.u32 base,base,idx;shr.u32 tmp,low,7;add.u32 base,base,tmp;shl.b32 bits,base,7;\nshr.u32 tmp,v,2;and.b32 tmp,tmp,1;shl.b32 tmp,tmp,15;or.b32 bits,bits,tmp;and.b32 low,low,127;or.b32 bits,bits,low;\nld.global.nc.u32 off,[root+36];cvt.u64.u32 ptr,off;add.u64 ptr,root,ptr;mul.wide.u32 adr,unit,8;add.u64 ptr,ptr,adr;\nld.global.nc.u32 pres,[ptr];ld.global.nc.u32 start,[ptr+4];and.b32 tmp,pres,bit;setp.eq.u32 p,tmp,0;@p bra DC_FINISH;\nsub.u32 tmp,bit,1;and.b32 tmp,tmp,pres;popc.b32 rank,tmp;add.u32 start,start,rank;\nld.global.nc.u32 off,[root+40];cvt.u64.u32 ptr,off;add.u64 ptr,root,ptr;mul.wide.u32 adr,start,8;add.u64 ptr,ptr,adr;\nld.global.nc.u32 mask,[ptr];ld.global.nc.u32 pos,[ptr+4];shl.b32 bit,1,ix;and.b32 tmp,mask,bit;setp.eq.u32 p,tmp,0;@p bra DC_FINISH;\nsub.u32 tmp,bit,1;and.b32 tmp,tmp,mask;popc.b32 rank,tmp;add.u32 pos,pos,rank;\nld.global.nc.u32 off,[root+44];cvt.u64.u32 ptr,off;add.u64 ptr,root,ptr;mul.wide.u32 adr,pos,2;add.u64 ptr,ptr,adr;ld.global.nc.u16 bits,[ptr];\nDC_FINISH:mov.u32 $0,bits;\n}',constraints='=r,l,r,r,r',args=[BANK,r.to(tl.uint32),c.to(tl.uint32),K],dtype=tl.uint32,is_pure=True,pack=1)
    return bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)


@triton.jit
def dc_unpack_kernel(BANK,OUT,N:tl.constexpr,K:tl.constexpr,TRANS:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);r=i//K;c=i%K
    v=dc_load(BANK,tl.minimum(r,N-1),c,K)
    tl.store(OUT+(c*N+r if TRANS else i),v,i<N*K)


def dc_unpack(bank,out,trans=False):
    n,k=bank['N'],bank['K'];dc_unpack_kernel[(triton.cdiv(n*k,256),)](bank['coded'],out,N=n,K=k,TRANS=trans,B=256,launch_pdl=True)
    return out
