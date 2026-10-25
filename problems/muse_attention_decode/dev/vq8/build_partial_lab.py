"""Build paired final-S11 vs direct projection-partials port lab from the shared attention fixture."""
from pathlib import Path
import partial_port
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/vq8_layer.py').read_text()
s=s[:s.index('def lab(deadline, weights):')]+s[s.index('def lab(deadline, weights):'):]
s=s.replace('    with ThreadPoolExecutor(4) as pool:', '    with ThreadPoolExecutor(4) as pool:')
a=s.index('        port = pool.submit(_attn_port_ext)', s.index('def lab(deadline, weights):'))
b=s.index("    lines.append('COMPILE_S",a)
s=s[:a]+'''        port = pool.submit(_attn_port_ext)
        old = pool.submit(_vq8_build, False, True)
        pp = [pool.submit(partial_port_ext, fast) for fast in (False, True)]
        for f in pp: f.result()
        _VQ8_OLD = old.result()
        port.result()
'''+s[b:]
s=s.replace("_vq8_ext().vq8_resources(4)", "[partial_port_ext().occupancy(1),partial_port_ext().occupancy(8)]")
s=s.replace('attention_vq8(qkv, data, (4, 1))', 'attention_partial_port(qkv, data)')
a=s.index('                for st in ((4, 1), (6, 8)):');b=s.index('                stub_us',a)
s=s[:a]+'''                for fast in (False,True):
                    kernels.append(('partial_fast' if fast else 'partial',
                                    lambda d,fast=fast:attention_partial_port(qkv,d,fast),fast))
'''+s[b:]
a=s.index('                if seed == 4100 and');b=s.index('            except Exception as exc:',a)
s=s[:a]+s[b:]
s=s.replace('META paired AB/BA x8, 3 seeds;', 'META direct-partials attn_main / fast sincos vs final S11, paired AB/BA x8, 3 seeds;')
s=s.replace('local FA3 stub fails by design.', 'all arms checked against TASK reference on the same stored K/V.')
# Insert the extension before lab, without any runtime local imports.
a=s.index('def lab(deadline, weights):')
s=s[:a]+partial_port.cell()+'\n\n'+s[a:]
p=ROOT/'prof/lab/attn_partial_port.py';p.write_text(s);print(p,len(s))
