"""PERMW (b88d04fe) onto S12's F4+wide nf cell. Usage: INPUT OUTPUT.

Only the nf CUDA and packed-scale prepare transform change. The input can be
S12 final or its a92b6619 precursor; all matching is scoped to the nf cell.
"""
import ast
import hashlib
import runpy
import sys
from pathlib import Path


def once(s,a,b):
    assert s.count(a)==1,a[:120]
    return s.replace(a,b)


def cuda_patch(c):
    assert 'F(26,32,true,7,0,0,0)' in c and 'F(21,32,false,6,0,0,0)' in c
    c=once(c,'void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){',
           'void dq(unsigned a,unsigned b,unsigned raw,unsigned*out,unsigned scale_base){')
    old=''' unsigned s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u,s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;
 unsigned s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u,s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;'''
    new=''' unsigned even,odd,rot=rr(raw,8);
 asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(even):"r"(raw),"r"(scale_base));
 asm("lop3.b32 %0,%1,0x0ff00ff0,%2,0xea;":"=r"(odd):"r"(rot),"r"(scale_base));
 unsigned s0=__byte_perm(even,even,0x1010),s2=__byte_perm(even,even,0x3232);
 unsigned s1=__byte_perm(odd,odd,0x1010),s3=__byte_perm(odd,odd,0x3232);'''
    c=once(c,old,new)
    c=once(c,'void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){',
           'void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b,uint64_t desc){')
    c=once(c,'uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);','uint64_t d=desc;')
    c=once(c,'void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){',
           'void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs,uint64_t desc){')
    c=once(c,'mma<NT>(c,a+b*4,xs,b);','mma<NT>(c,a+b*4,xs,b,desc);')
    c=once(c,'float*Y,float*SS,int M){','float*Y,float*SS,int M,unsigned scale_base){')
    c=once(c,'int t=threadIdx.x,w=t>>5,l=t&31;\n int total=',
           'int t=threadIdx.x,w=t>>5,l=t&31;\n uint64_t desc0=(uint64_t)((smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n int total=')
    # The older generic branch is not instantiated by S12, but keep it compilable.
    c=c.replace('dq(q.x,q.y,sub?sv.z:sv.x,af[sub]);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8);',
                'dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);')
    c=c.replace('dq(q.x,q.y,sf,af[sub]+g*8);','dq(q.x,q.y,sf,af[sub]+g*8,scale_base);')
    c=c.replace('mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);',
                'mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64,desc0+(st*NT*KT+sub*NT*64)/8);')
    c=once(c,'>>>(BR_ARGS(v),(int)M);','>>>(BR_ARGS(v),(int)M,0x70007000u);')
    return c


def patch(text):
    n=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CUDA')
    if 'F(26,32,true,3,0,0,0)' in ast.literal_eval(n.value):
        # Final S12 dropped wide after its second whole-layer test. PERMW needs
        # that packed layout; restore it as part of this concrete proposal.
        dependency=Path(__file__).with_name('patch_s11_wide.py')
        assert hashlib.sha256(dependency.read_bytes()).hexdigest()=='b2748e2359f892b679be32eaf6e8fff4d8bba5f6df225c1e5db35b66b4dded18'
        text=runpy.run_path(str(dependency))['patch'](text)
        n=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CUDA')
    cu=cuda_patch(ast.literal_eval(n.value))
    lines=text.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['NF_CUDA='+repr(cu)+'\n'];text=''.join(lines)
    anchor="state['s5_hh_base']=nf_wide(issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm']))))"
    text=once(text,anchor,anchor.replace('=nf_wide(', '=nf_perm(nf_wide(')+')')
    helper='''def nf_perm(bank):
    # Rotate the same packed u8 scale bits; no decoded weight/scale values stored.
    for k in ('w13_sf','w2_sf'):
        sf=bank[k];v=sf.contiguous().reshape(-1).view(torch.int32).to(torch.int64)&0xffffffff
        bank[k]=(((v<<4)|(v>>28))&0xffffffff).to(torch.int32).view(torch.uint8).reshape(sf.shape)
    return bank

'''
    text=once(text,'def nf_plan(data,base,native,streams):',helper+'def nf_plan(data,base,native,streams):')
    compile(text,'<S12 PERMW>','exec')
    return text


if __name__=='__main__':
    src,dst=map(Path,sys.argv[1:]);dst.write_text(patch(src.read_text()))
    print(dst,hashlib.sha256(dst.read_bytes()).hexdigest())
