"""Apply f976b09d's LIW layout onto S11 final (also accepts its F4 edit).

Usage: python3 patch_s11_wide.py INPUT OUTPUT
Reorders the same packed code/scale bytes; no precision expansion in prepare.
"""
import ast
import hashlib
import sys
from pathlib import Path


def once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def patch(text):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == 'NF_CUDA' for t in n.targets))
    cu = ast.literal_eval(node.value)
    cu = once(cu, 'F(26,32,true,3,0,0,0) F(21,32,false,2,0,0,0)',
              'F(26,32,true,7,0,0,0) F(21,32,false,6,0,0,0)')
    cu = once(cu, 'constexpr int BS=((S*(QB+SB)+1023)&~1023);',
              'constexpr bool WIDE=VEC>=6;constexpr int BS=((S*(QB+SB)+1023)&~1023);')
    cu = once(cu, 'if(ws){unsigned i=t-64,g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);}',
              'if(ws){unsigned i=t-64;if constexpr(WIDE)sg=SC+(((unsigned)e*(N/16)+mt*4+i/8)*128*32+(i%8)*16);else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sg=SC+((((unsigned)e*(N/16)+mt*4+gp)*128+g)*32+hf*16);}}')
    cu = once(cu, 'unsigned qd=smaddr(qs+t*16),sd=0;',
              'unsigned qd=smaddr(qs+(WIDE?(t>>4)*512+(t&15)*16:t*16)),sd=0;')
    cu = once(cu, 'if(ws){unsigned i=t-64,g=i/8,gp=(i%8)/2,hf=i%2;sd=smaddr(ss+g*128+gp*32+hf*16);}',
              'if(ws){unsigned i=t-64;if constexpr(WIDE)sd=smaddr(ss+i*16);else{unsigned g=i/8,gp=(i%8)/2,hf=i%2;sd=smaddr(ss+g*128+gp*32+hf*16);}}')
    cu = once(cu, 'qd+slot*QB+g*1024', 'qd+slot*QB+(WIDE?(g>>1)*2048+(g&1)*256:g*1024)')
    cu = once(cu, 'if constexpr(VEC<4)asm volatile', 'if constexpr(VEC<4||WIDE)asm volatile')
    old = '''     #pragma unroll
     for(int sub=0;sub<2;++sub){
      #pragma unroll
      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);
       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}'''
    new = '''     uint4 sv;if constexpr(WIDE)sv=*(const uint4*)(ss+st*SB+(w*8+(l>>2))*16);
     #pragma unroll
     for(int sub=0;sub<2;++sub){
      if constexpr(WIDE){uint4 q=*(const uint4*)(qs+st*QB+sub*2048+t*16);
       dq(q.x,q.y,sub?sv.z:sv.x,af[sub]);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8);
      }else{
      #pragma unroll
      for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);
       unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}}'''
    cu = once(cu, old, new)
    cu = once(cu, 'FIRST&&(VEC==3||VEC==5)', 'FIRST&&(VEC==3||VEC==5||VEC==7)')
    lines = text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['NF_CUDA='+repr(cu)+'\n']
    text = ''.join(lines)
    text = once(text, "state['s5_hh_base']=issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm'])))",
                "state['s5_hh_base']=nf_wide(issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm']))))")
    helper = '''def nf_wide(bank):
    for k in ('w13','w2'):
        q,sf=bank[k],bank[k+'_sf'];e,ng=sf.shape[:2]
        bank[k]=q.view(torch.uint8).reshape(e,ng,64,2,32,8).transpose(3,4).contiguous().view(q.dtype).reshape(q.shape)
        bank[k+'_sf']=sf.reshape(e,ng,32,4,8,4).transpose(3,4).contiguous().reshape(sf.shape)
    return bank

'''
    text = once(text, 'def nf_plan(data,base,native,streams):', helper+'def nf_plan(data,base,native,streams):')
    compile(text, '<S11 LIW>', 'exec')
    return text


if __name__ == '__main__':
    src,dst=map(Path,sys.argv[1:]);dst.write_text(patch(src.read_text()))
    print(dst,hashlib.sha256(dst.read_bytes()).hexdigest())
