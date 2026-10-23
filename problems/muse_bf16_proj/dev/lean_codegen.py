"""Generate a Torch-header-free ABI while preserving the selected GPU kernel bodies."""
import ast
from pathlib import Path
import re
P=Path(__file__).resolve().parents[1]


def values_of(text):
    out={}
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,(ast.Constant,ast.Tuple,ast.List,ast.Dict)):
            out[n.targets[0].id]=ast.literal_eval(n.value)
    return out


def device_only(source,namespace):
    source=source[:source.index('static PFN_cuTensorMapEncodeTiled_v12000 encode_fn()')]
    return source+'\n} // namespace '+namespace+'\n'


def convert(text):
    values=values_of(text)
    skin=values['CUDA_SRC'];end=re.search(r'template <int NT, int TR, int U, int NA, int XP, int LD>\nvoid launch_t',skin).start()
    skin=skin[:end]+'\n} // namespace muse_proj\n'
    skin=skin.replace('#include <torch/extension.h>','').replace('#include <ATen/cuda/CUDAContext.h>','').replace('namespace cg = cooperative_groups;','')
    v2='PAIR_SRC' in values
    pair=device_only(values['PAIR_SRC'] if v2 else values['HOP_SPECIALIZED'],'muse_pair')
    configs=values['FUSED_CONFIGS']
    specs=sorted({(c[0],c[1],c[2],c[5],layer=='router',c[3]==3) for (layer,m),row in configs.items() for c in row})
    instances=values['INSTANCES']
    skin_cases=' '.join('LSKIN('+','.join(map(str,c))+')' for c in instances)
    pair_cases=' '.join(f'LPAIR({bt},{bk},{st},{th},{str(of).lower()},{str(hh).lower()})' for bt,bk,st,th,of,hh in specs)
    legacy='';old_specs=[]
    if v2:
        old_specs=sorted({(c[0],c[1],c[2],layer=='router') for (layer,m),c in values['H200_WINNERS'].items() if c and c[0]!='lt'})
        if old_specs:
            ns={**values}
            for name in ('_wgmma_fn','_hop_source'):
                node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
                exec(ast.get_source_segment(text,node),ns)
            legacy=device_only(ns['_hop_source'](),'muse_wg').replace('muse_wg','muse_legacy')
    assert 'torch::' not in skin+pair+legacy and 'TORCH_CHECK' not in skin+pair+legacy
    raw=(P/'dev/lean_host.cuh').read_text().replace('LEAN_DEVICE_SKINNY',skin+legacy).replace('LEAN_DEVICE_PAIR',pair).replace('LEAN_SKIN_CASES',skin_cases).replace('LEAN_PAIR_CASES',pair_cases)
    if old_specs:
        old_cases=' '.join(f'LOLD({bt},{bk},{st},{str(of).lower()})' for bt,bk,st,of in old_specs)
        raw+='\n'+(P/'dev/lean_old_host.cuh').read_text().replace('LEAN_OLD_CASES',old_cases)
    skip={'CUDA_SRC','CPP_SRC','HOP_SPECIALIZED','HOP_HELPERS','HOP_MAP','HOP_KERNEL','HOP_CONFIGS','PAIR_SRC','LT_SRC','LT_NN_SRC'}
    skip_defs={'_source','_hop_source','_wgmma_fn','_module'}
    parts=[]
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in skip for t in node.targets):continue
        if isinstance(node,ast.FunctionDef) and node.name in skip_defs:continue
        parts.append(ast.get_source_segment(text,node))
    source='# okc: problem=muse_bf16_proj\n'+'\n\n'.join(parts)+'\n\n'+(P/'dev/lean_module.py').read_text()+"\nRAW_CUDA = r'''"+raw+"'''\n"
    if v2:
        source=source.replace("state['mod'].hopper(w,x,out,*scratch,bt,bk,st,mode,split)","state['mod'].old_hopper(w,x,out,*scratch,bt,bk,st,mode,split)")
        source=source.replace("state['mod'].hopper_pair(w,x,out,*scratch,*config[1:])","state['mod'].hopper(w,x,out,*scratch,*config[1:])")
    return source,{'new_fused_specializations':len(specs),'old_specializations':len(old_specs),'small_specializations':len(instances)}
