"""Attention-board validation when every whole-layer host quota is exhausted."""
import ast,hashlib,importlib.util,re
from pathlib import Path
import build_global
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'runs/muse_attention_decode/20261025T030917Z_submit_81cccbc19410433b.py').read_text()
a=s.rindex('if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:');s=s[:a]
# The frozen incumbent's import-only search is replaced by the measured shape dispatch below.
# Keep only definitions/initial state; no import tuning or hidden synthetic work.
tree=ast.parse(s);lines=s.splitlines(True)
for n in reversed(tree.body):
    if isinstance(n,ast.If) and '_lab' in ast.get_source_segment(s,n):lines[n.lineno-1:n.end_lineno]=[]
s=''.join(lines)
def driver_cell(path,prefix,cuda,cpp=None):
    d=path.read_text();nodes=ast.parse(d).body
    cell='\n'.join(ast.get_source_segment(d,n) for n in nodes if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('CPP','VQ8_CPP','_MOD','_PLANS','_FREQ','_COUNTERS','_EXT','_SMS','_INVF') for t in n.targets)))
    names=('CUDA','CPP','_MOD','_PLANS','_FREQ','_COUNTERS','module','plan','attention') if prefix=='gc_' else ('VQ8_CUDA','VQ8_CPP','_EXT','_SMS','_INVF','vq8_ext','_num_sms','_inv_freq','vq8_supported','vq8_attention')
    for name in names:cell=re.sub(r'\b'+name+r'\b',prefix+name,cell)
    return prefix+('CUDA' if prefix=='gc_' else 'VQ8_CUDA')+'='+repr(cuda)+'\n'+(prefix+'CPP='+repr(cpp)+'\n' if cpp else '')+cell+'\n'
cuda=build_global.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
s+='\nimport hashlib,re\n'+driver_cell(HERE/'global_driver.py','gc_',cuda,cpp)
p=HERE.parent/'full128/build.py';q=importlib.util.spec_from_file_location('_p5_board_vq_source',p);v=importlib.util.module_from_spec(q);q.loader.exec_module(v)
s+='\n'+driver_cell(HERE.parent/'full_head/driver.py','p8_',v.source())
# p8 is the two-head winner: use original pair allocator/grid, not the experimental head driver.
s=s.replace('G = grid or 2*_num_sms(device)','G = grid or _num_sms(device)') if False else s
s=s.replace('G = grid or 2*p8__num_sms(device)','G = grid or p8__num_sms(device)').replace('NS = B * HK\n','NS = B * (HK // 2)\n')
s+='''

def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg=data['config']
    if cfg['q_len']==1:return gc_attention(data)
    if not cfg['window'] and cfg['batch'] in (16,32) and cfg['num_kv_heads']==2:
        return p8_vq8_attention(data,stages=4,min_tiles=8)
    return _BASE_FN(data)
'''
compile(s,'p5_global_board.py','exec');p=HERE/'out/p5_global_board.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
