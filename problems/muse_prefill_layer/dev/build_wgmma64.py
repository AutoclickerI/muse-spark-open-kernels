from pathlib import Path
import ast,re
p=Path('problems/muse_prefill_layer')
s=(p/'cand_v3.py').read_text()
line=next(x for x in s.splitlines() if x.startswith('CUDA_SRC = '))
c=ast.literal_eval(line.split('=',1)[1].strip())
c=c.replace('constexpr int BN = 128;', 'constexpr int BN = 64;')
c=c.replace('float (&acc)[64]', 'float (&acc)[32]').replace('float (&d)[64]', 'float (&d)[32]').replace('float acc[2][64]', 'float acc[2][32]').replace('i < 64; ++i', 'i < 32; ++i')
start=c.index('  asm volatile(\n',c.index('uint64_t desc = make_desc_b'))
end=c.index('\n#else',start)
reg=', '.join('%'+str(i) for i in range(32))
out=',\n        '.join(', '.join('"+f"(d[%d])'%i for i in range(j,j+8)) for j in range(0,32,8))
c=c[:start]+f'''  asm volatile(
      "{{\\n.reg .pred p;\\nsetp.ne.b32 p, %37, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{{{reg}}}, {{%32, %33, %34, %35}}, %36, p, 1, 1, 0;\\n}}\\n"
      : {out}
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));'''+c[end:]
s=s.replace(line,'CUDA_SRC = '+repr(c)).replace('BN = 128          # tokens per tile','BN = 64           # tokens per tile').replace('name="muse_prefill_grouped_w4a16"','name="muse_prefill_grouped_w4a16_bn64"')
(p/'cand_v7.py').write_text(s)
(p/'dev/moe_w4a16_bn64.cu').write_text(c)
