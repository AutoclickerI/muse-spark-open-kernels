"""Pair direct-partials S2/S3/S4 sliding gather rings against frozen S11."""
from pathlib import Path
import port_stages
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
src=(ROOT/'prof/lab/attn_partial_port.py').read_text()
src=src.replace('META direct-partials attn_main / fast sincos','META direct-partials sliding gather S2/S3/S4')
src=src.replace('        pp = [pool.submit(partial_port_ext, fast) for fast in (False, True)]',
    '        pp = [pool.submit(fn) for fn in (partial_port_ext,partial_port_ext_s2,partial_port_ext_s3)]')
a=src.index('                for fast in (False,True):');b=src.index('                stub_us',a)
src=src[:a]+'''                for name,fn in [('s4',attention_partial_port),('s3',attention_partial_port_s3),('s2',attention_partial_port_s2)]:
                    kernels.append((name,lambda d,fn=fn:fn(qkv,d),None))
'''+src[b:]
a=src.index('    order = ((');b=src.index('    for seed in (4100, 5100, 6100):',a)
order=tuple((b,'sliding',1) for b in (1,16,32))+((16,'full',1),(32,'full',1))
src=src[:a]+'    order = '+repr(order)+'\n'+src[b:]
a=src.index('def lab(deadline, weights):')
src=src[:a]+port_stages.cell(2)+'\n'+port_stages.cell(3)+'\n'+src[a:]
p=ROOT/'prof/lab/attn_partial_stages.py';p.write_text(src);print(p,len(src))
