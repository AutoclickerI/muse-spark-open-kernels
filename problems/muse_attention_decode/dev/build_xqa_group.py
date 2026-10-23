"""Hopper WGMMA swap-AB path with the same executable MMA stand-in on sm_120."""
from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]
source=(root/'cand_xqa_pipeline.py').read_text().split('\n_PIPE_PLAN = {}')[0]
tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='X_CUDA' for t in n.targets))
c=(root/'dev/xqa_group.cu').read_text()
wrappers=[]
for name,op,extra in [('wg8','m64n64k32.f32.e4m3.e4m3',''),('wg16','m64n64k16.f32.f16.f16',',1,0')]:
 regs=','.join('%'+str(i) for i in range(32));outputs=','.join(f'"+f"(d[{i//4}][{i%4}])' for i in range(32))
 wrappers.append(f'''DI void {name}(float (&d)[8][4],uint64_t a,uint64_t b){{
#if USE_WGMMA
 asm volatile("{{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.{op} {{{regs}}},%32,%33,p,1,1{extra};}}":{outputs}:"l"(a),"l"(b));
#endif
}}''')
c=c.replace('WG_WRAPPERS','\n'.join(wrappers))
(root/'dev/xqa_group_generated.cu').write_text(c)
lines=source.splitlines(True);lines[node.lineno-1:node.end_lineno]=['X_CUDA = r"""\n',c,'\n"""\n']
text=''.join(lines).replace('muse_xqa_pipeline_sm','muse_xqa_group_sm')
text=text.replace("arch=f'{major}{minor}'", "arch='90a' if major==9 else f'{major}{minor}'")
text=text.replace('X_STAGES = 3','X_STAGES = 5').replace('(maxkeys+127)//128','(maxkeys+63)//64')
(root/'dev/xqa_group_candidate.py').write_text(text)
print('Built group candidate')
