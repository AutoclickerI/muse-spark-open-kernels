"""Keep the fused math, replace Torch-heavy C++ wrappers with a checked raw-pointer ABI."""
import ast
from pathlib import Path
import re
import sys
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'dev'))
import build_portfolio_v2 as builder
text=(P/'cand_fused.py').read_text();values=builder.assignments(text)
skin=values['CUDA_SRC'];end=re.search(r'template <int NT, int TR, int U, int NA, int XP, int LD>\nvoid launch_t',skin).start()
skin=skin[:end]+'\n} // namespace muse_proj\n'
skin=skin.replace('#include <torch/extension.h>','').replace('#include <ATen/cuda/CUDAContext.h>','').replace('namespace cg = cooperative_groups;','')
pair=values['HOP_SPECIALIZED'];pair=pair[:pair.index('static PFN_cuTensorMapEncodeTiled_v12000 encode_fn()')]+'\n} // namespace muse_pair\n'
assert 'torch::' not in skin+pair and 'TORCH_CHECK' not in skin+pair
configs=values['FUSED_CONFIGS'];specs=sorted({(c[0],c[1],c[2],c[5],layer=='router',c[3]==3) for (layer,m),row in configs.items() for c in row})
instances=values['INSTANCES'];skin_cases=' '.join('LSKIN('+','.join(map(str,c))+')' for c in instances)
pair_cases=' '.join(f'LPAIR({bt},{bk},{st},{th},{str(of).lower()},{str(hh).lower()})' for bt,bk,st,th,of,hh in specs)
raw=(P/'dev/lean_host.cuh').read_text().replace('LEAN_DEVICE_SKINNY',skin).replace('LEAN_DEVICE_PAIR',pair).replace('LEAN_SKIN_CASES',skin_cases).replace('LEAN_PAIR_CASES',pair_cases)
skip={'CUDA_SRC','CPP_SRC','HOP_SPECIALIZED','LT_SRC','LT_NN_SRC'}
skip_def={'_source','_hop_source','_module'}
parts=[]
for node in ast.parse(text).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in skip for t in node.targets):continue
    if isinstance(node,ast.FunctionDef) and node.name in skip_def:continue
    parts.append(ast.get_source_segment(text,node))
source='# okc: problem=muse_bf16_proj\n'+'\n\n'.join(parts)+'\n\n'+(P/'dev/lean_module.py').read_text()+"\nRAW_CUDA = r'''"+raw+"'''\n"
source=source.replace("print(f'FUSED ","print(f'FUSED_LEAN ")
out=P/'cand_fused_lean.py';compile(source,str(out),'exec');out.write_text(source);print('lean source bytes',len(source),'GPU specializations',len(specs)+len(instances))
