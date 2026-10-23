"""Build K64/K128 gather/occupancy variants from a frozen K128 source.

Sent files are inputs only. The new source specializes full attention to remove
RoPE code, and tests a K64 gather with up to seven stages and two CTAs per SM.
"""
from pathlib import Path
import ast

root = Path(__file__).resolve().parents[1]
source = (root / 'cand_xqa_k128.py').read_text()
tree = ast.parse(source)
assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == 'X_CUDA' for t in n.targets))
c = ast.literal_eval(assignment.value)
c = c.replace('constexpr int KT=128,KB=KT*64;', '#ifndef XKEYS\n#define XKEYS 64\n#endif\nconstexpr int KT=XKEYS,KB=KT*64;')
c = c.replace('template<int ST,bool P8>', 'template<int ST,bool P8,bool ROPE>')
c = c.replace('__launch_bounds__(256,1)', '__launch_bounds__(256,2)')
c = c.replace('if(p.freq){', 'if constexpr(ROPE){')
c = c.replace('xqa<ST,P8>', 'xqa<ST,P8,ROPE>')
# The gather has 256 threads: 64 rows per loop, for K64 or K128.
start = c.index(' if(stages==5){if(p8)go<5,true>')
end = c.index(' auto err=cudaGetLastError();', start)
dispatch = []
for st in (7, 5, 4, 3):
    dispatch.append((' if' if st == 7 else ' else if') + f'(stages=={st})' + '{')
    for p8 in (True, False):
        dispatch.append(('  if' if p8 else '  else') + ('(p8)' if p8 else '') + '{')
        dispatch.append(f'   if(p.window)go<{st},{str(p8).lower()},true>(p,s);else go<{st},{str(p8).lower()},false>(p,s);')
        dispatch.append('  }')
    dispatch.append(' }')
c = c[:start] + '\n'.join(dispatch) + '\n' + c[end:]
(root / 'dev/xqa_pipeline.cu').write_text(c)
# Keep original Python glue as a source template, without publishing or running
# its old 24-way prepare. The new packed diagnostic is built after profiling.
cut = source.index('\n_W8_PLAN = {}')
text = source[:cut]
old_cuda = source[source.index('X_CUDA = r"""'):source.index('\nX_CPP = r"""')]
text = text.replace(old_cuda, 'X_CUDA = r"""\n' + c + '\n"""')
text = text.replace('muse_xqa_w8_sm', 'muse_xqa_pipeline_sm')
text = text.replace('X_STAGES = 3', 'X_STAGES = 5')
text = text.replace('(maxkeys+127)//128', '(maxkeys+63)//64')
(root / 'dev/xqa_pipeline_candidate.py').write_text(text)
text128 = text.replace('#define XKEYS 64', '#define XKEYS 128')
text128 = text128.replace('muse_xqa_pipeline_sm', 'muse_xqa_pipeline128_sm').replace('X_STAGES = 5', 'X_STAGES = 3')
text128 = text128.replace('(maxkeys+63)//64', '(maxkeys+127)//128')
(root / 'dev/xqa_pipeline128_candidate.py').write_text(text128)
print('Built', root / 'dev/xqa_pipeline_candidate.py')
