"""Add measured WGMMA T16 rows without replacing integration's incumbent T1 codec."""
import ast
import runpy
from pathlib import Path

H = Path(__file__).resolve().parent
runpy.run_path(str(H / 'build_wg.py'))
tree = ast.parse((H / 'wg_body.py').read_text())
names = {}
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name.startswith(('cd_', '_cd_')):
        names[n.name] = n.name.replace('cd_', 'wcd_', 1)
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name) and t.id.startswith(('CD_', '_CD_')):
                names[t.id] = t.id.replace('CD_', 'WCD_', 1)

class Rename(ast.NodeTransformer):
    def visit_Name(self, n):
        if n.id in names:n.id = names[n.id]
        return n
    def visit_FunctionDef(self, n):
        self.generic_visit(n)
        if n.name in names:n.name = names[n.name]
        return n

body = ast.unparse(Rename().visit(tree)) + '\n'
old = 'WCD_INSTANCES = tuple(((nt, rt, pipe, v) for v in (3, 4) for nt in (1, 2, 4) for rt in (1, 2) for pipe in (2, 4)))'
assert body.count(old) == 1
body = body.replace(old, 'WCD_INSTANCES = ((1, 1, 2, 3),)')
scaffold = '''
_WGC_ROWS = __ROWS__

def _wgc_module():
    wcd_module()
    wg_module()

def _wgc_prepare(state):
    state['wgmma_coded']={}
    for t,name in _WGC_ROWS:
        if name in state['wgmma_coded']:continue
        w=state[name]
        c=wg_encode(w,kmajor=True)
        if not torch.equal(wg_decode(c).view(torch.int16),w.view(torch.int16)):
            raise RuntimeError(f'WGMMA {name}: bit-exact decode failed')
        state['wgmma_coded'][name]=c

def _wgc_force(state):
    for t,name in _WGC_ROWS:state['proj_plan'][(t,name)]='custom'

def _wgc_project(x,name,state):
    NW,S,WB=_WGC_ROWS[(x.shape[0],name)]
    return wg_gemv(x,state['wgmma_coded'][name],S,NW=NW,WB=WB,prew=1,pdl=PDL)
'''
# Reuse the already screened additive insertion points, with independent symbols/state.
prototype = ast.parse((H / 'patch_direct_qkv3.py').read_text())
rows = {(16, 'qkv_gate'): (8, 16, 2), (16, 'unpooler'): (8, 4, 2)}
for n in prototype.body:
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
        n.value.value = 'Measured register-A WGMMA T16 handoff (04d89c7d); additive onto S12/S13.\nT32 unpooler excluded: its initial win did not repeat robustly. No T1 replacement.'
    if isinstance(n, ast.Assign) and n.targets[0].id in ('BODY', 'SCAFFOLD', 'ROWS'):
        n.value = ast.parse(repr({'BODY': body, 'SCAFFOLD': scaffold, 'ROWS': rows}[n.targets[0].id]), mode='eval').body
    if isinstance(n, ast.FunctionDef) and n.name == 'patch':
        text = r'''def patch(src, rows=None):
    rows = ROWS if rows is None else rows
    assert '_WGC_ROWS = ' not in src
    marker = '# Projection portfolio:'
    assert src.count(marker) == 1
    src = src.replace(marker, BODY + '\n' + SCAFFOLD.replace('__ROWS__', repr(rows)) + '\n' + marker, 1)
    old = 'def _project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):\n'
    assert src.count(old) == 1
    src = src.replace(old, old + '    if (x.shape[0],name) in _WGC_ROWS:\n        return _wgc_project(x,name,state)\n', 1)
    match = re.search(r'with ThreadPoolExecutor\(max_workers=(\d+)\) as native_pool:', src)
    assert match
    src = src[:match.start()] + f'with ThreadPoolExecutor(max_workers={int(match.group(1))+1}) as native_pool:\n        wgc_future=native_pool.submit(_wgc_module)' + src[match.end():]
    old = '        state=_prepare_weights(weights)\n'
    assert src.count(old) == 1
    src = src.replace(old, old + '        wgc_future.result();_wgc_prepare(state)\n', 1)
    old = '    _gate_split_prepare(state,device)\n'
    assert src.count(old) == 1
    src = src.replace(old, old + '    _wgc_force(state)\n', 1)
    compile(src, '<WGMMA row>', 'exec')
    return src
'''
        n2 = ast.parse(text).body[0]
        prototype.body[prototype.body.index(n)] = n2
out = ast.unparse(prototype) + '\n'
compile(out, '<WGMMA patch>', 'exec')
(H / 'patch_wg_t16.py').write_text(out)
print(H / 'patch_wg_t16.py')
