"""S7: frozen S6 with H_H's heavy WGMMA path reading the packed uint8 scale codes.

S6's `issue_word_bank` stored every block scale as its BF16 bit pattern ((code<<4)+0x7000, int16) in prepare:
a widened, partly decoded scale bank. S7 deletes it; the idist kernel copies the uint8 codes (half the bytes)
and forms the same BF16 word in registers (byte_perm, *16, +0x7000), so the arithmetic is unchanged.
Nothing else in S6 changes.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
BASE = R / 'problems/muse_decode_layer/cand_r3_best7_s6.py'
BASE_SHA = '4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f'

CUDA_EDITS = (
    ('__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned long long raw,unsigned*out){\n'
     ' unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));\n'
     ' unsigned s0=__byte_perm((unsigned)raw,0u,0x1010u);\n'
     ' unsigned s1=__byte_perm((unsigned)raw,0u,0x3232u);\n'
     ' unsigned s2=__byte_perm((unsigned)(raw>>32),0u,0x1010u);\n'
     ' unsigned s3=__byte_perm((unsigned)(raw>>32),0u,0x3232u);\n',
     '__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){\n'
     ' unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));\n'
     ' // uint8 code c -> BF16 bits (c<<4)+0x7000 in both halves, formed in registers.\n'
     ' unsigned s0=__byte_perm(raw,0u,0x4040u)*16u+0x70007000u;\n'
     ' unsigned s1=__byte_perm(raw,0u,0x4141u)*16u+0x70007000u;\n'
     ' unsigned s2=__byte_perm(raw,0u,0x4242u)*16u+0x70007000u;\n'
     ' unsigned s3=__byte_perm(raw,0u,0x4343u)*16u+0x70007000u;\n', 1),
    ('if(l<8)async_copy(s+g*256+w*64+l*8,S+z*64+l*8,8,1);',
     'if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);', 1),
    ('constexpr int QB=64*KT/2,SB=64*KT/8,', 'constexpr int QB=64*KT/2,SB=64*KT/16,', 1),
    ('unsigned long long sf=*(const unsigned long long*)(ss+st*SB+z*256+w*64+(l>>2)*8);dq(q.x,q.y,sf,af[reg]+g*8);',
     'unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[reg]+g*8);', 1),
    ('1023+((S*(64*KT/2+64*KT/8)+1023)&~1023)', '1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)', 3),
)
PY_EDITS = (
    ("    if name=='hh_m128':return pm_call(data,state['s5_hh_base'],state['s5_hh_word'],state['moe_bank'],'H','H')",
     "    if name=='hh_m128':return pm_call(data,state['s5_hh_base'],state['moe_bank'],'H','H')", 1),
    ("    state['s5_hh_word']=issue_word_bank(state['s5_hh_base'])\n", '', 1),
    ("def issue_word_bank(bank):\n    word=dict(bank)\n    for key in ('w13_sf','w2_sf'):\n"
     "        word[key]=((bank[key].to(torch.int32)<<4)+0x7000).to(torch.int16)\n    return word\n", '', 1),
    ('def pm_plan(data,base,word,native,first=', 'def pm_plan(data,base,native,first=', 1),
    ("    word13=vector(word,1,tasks[0]);word2=vector(word,0,tasks[0])\n    big13=vector(word,1,tasks[1]);big2=vector(word,0,tasks[1])",
     "    word13=vector(base,1,tasks[0]);word2=vector(base,0,tasks[0])\n    big13=vector(base,1,tasks[1]);big2=vector(base,0,tasks[1])", 1),
    ('nv,vector(word,1,tasks[3]),vector(word,0,tasks[3])]', 'nv,vector(base,1,tasks[3]),vector(base,0,tasks[3])]', 1),
    ('def pm_call(data,base,word,native,first,second):\n    for fn in pm_plan(data,base,word,native,first,second)',
     'def pm_call(data,base,native,first,second):\n    for fn in pm_plan(data,base,native,first,second)', 1),
)


def literal(s, key):
    return next(n for n in ast.parse(s).body if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == key for t in n.targets))


def main():
    s = BASE.read_text()
    assert hashlib.sha256(s.encode()).hexdigest() == BASE_SHA
    node = literal(s, 'S5_HH_CUDA')
    cuda = ast.literal_eval(node.value)
    old_cuda = cuda
    for old, new, n in CUDA_EDITS:
        assert cuda.count(old) == n, (old, cuda.count(old))
        cuda = cuda.replace(old, new)
    assert 'unsigned long long' not in cuda
    lines = s.splitlines(keepends=True)
    assert node.lineno == node.end_lineno
    lines[node.lineno - 1] = 'S5_HH_CUDA=' + repr(cuda) + '\n'
    s = ''.join(lines)
    for old, new, n in PY_EDITS:
        assert s.count(old) == n, (old, s.count(old))
        s = s.replace(old, new)
    for word in ('issue_word_bank', 's5_hh_word', 'torch.int16', '0x7000).to('):
        assert word not in s, word
    out = BASE.with_name('cand_r3_best7_s7.py')
    out.write_text(s)
    sys.path.insert(0, str(R / 'tools'))
    from okc_run import rejected_words
    assert not rejected_words(s), rejected_words(s)
    m = dict(source=str(out.relative_to(R)), sha256=hashlib.sha256(s.encode()).hexdigest(), base=str(BASE.relative_to(R)),
             base_sha256=BASE_SHA, old_S5_HH_CUDA_sha256=hashlib.sha256(old_cuda.encode()).hexdigest(),
             new_S5_HH_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
             change='H_H heavy path reads packed uint8 scale codes; BF16 scale words formed in-kernel; issue_word_bank deleted')
    (H / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    print(m['source'], m['sha256'])


if __name__ == '__main__':
    main()
