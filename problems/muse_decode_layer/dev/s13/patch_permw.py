"""PERMW at M=256 only; M=128 retains S12's original bank and kernel (3dbcaccd)."""
import ast
import importlib.util
from pathlib import Path


def once(src, old, new):
    assert src.count(old) == 1, old[:120]
    return src.replace(old, new)


def patch(src):
    tree = ast.parse(src)
    old_cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], 'id', '') == 'NF_CUDA')
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'nf_module')
    old_module = ast.get_source_segment(src, node).replace('def nf_module():', 'def nf128_module():').replace('NF_CUDA', 'NF128_CUDA')
    for name in ('wide_frozen', 'permw_frozen'):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
        m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        src = m.patch(src)
    old = "state['s5_hh_base']=nf_perm(nf_wide(issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm'])))))"
    new = "state['s5_hh_base']=issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm'])))\n    state['s13_permw_base']=nf_perm(nf_wide(dict(state['s5_hh_base'])))"
    src = once(src, old, new)
    src = once(src, 'cap=m*8+32*128;mod=nf_module()', 'cap=m*8+32*128;mod=nf128_module() if m<=128 else nf_module()')
    src = once(src, "nf_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])",
               "nf_plan(data,state['s13_permw_base'] if data['hidden_states'].shape[0]>128 else state['s5_hh_base'],state['moe_bank'],state['nf_streams'])")
    src = once(src, '        nf_future=native_pool.submit(nf_module)',
               '        nf128_future=native_pool.submit(nf128_module)\n        nf_future=native_pool.submit(nf_module)')
    src = once(src, '        nf_future.result()', '        nf128_future.result();nf_future.result()')
    # Keep the original M128 module's own cached function and its exact packed-bank ABI.
    src += '\nNF128_CUDA = ' + repr(old_cuda) + '\n\n@functools.lru_cache(None)\n' + old_module + '\n'
    compile(src, '<S13 PERMW256>', 'exec')
    return src
