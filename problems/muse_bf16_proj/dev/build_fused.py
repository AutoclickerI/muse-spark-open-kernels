"""Build a compact fusion experiment using validated dtype/hierarchy specializations.
This is local experimental pruning, not the receipt-derived portfolio v2.
"""
import ast
from pathlib import Path
import sys
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'dev'))
import build_portfolio_v2 as builder
configs=builder.local_fused_choices()
for key,cfg in [(('router',128),(128,64,3,3,32,160)),(('router',256),(256,64,2,3,32,160))]:
    if cfg not in configs[key]:configs[key].append(cfg)
source,specs=builder.fused_source([(cfg,layer=='router') for (layer,m),row in configs.items() for cfg in row])
source=source.replace('void hopper_pair(', 'void hopper(')
s=(P/'cand_reduce_hier.py').read_text();a=s.index('def _wgmma_fn(');b=s.index('\ndef _hopper():',a)
s=s[:a]+"HOP_SPECIALIZED = r'''"+source+"'''\n\n\ndef _hop_source():\n    return HOP_SPECIALIZED\n\n"+s[b:]
a=s.index('def _variants(');b=s.index('\n\ndef prepare(',a)
s=s[:a]+'''def _variants(m,n,k,sm_count):
    layer='router' if n==256 else 'qkv_gate' if n==2304 else None
    yield from FUSED_CONFIGS.get((layer,m),())
''' + s[b:]
s=s.replace(' + LT_SRC',' + LT_SRC + LT_NN_SRC')
s=s.replace('torch::Tensor lt_options(torch::Tensor W, torch::Tensor X, torch::Tensor Out);','''torch::Tensor lt_nn_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out);
void lt_nn_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index);
torch::Tensor lt_options(torch::Tensor W, torch::Tensor X, torch::Tensor Out);''')
s=s.replace('functions=["skinny", "hopper", "lt_options", "lt_run"]','functions=["skinny", "hopper", "lt_options", "lt_run", "lt_nn_options", "lt_nn_run"]')
s=s.replace("    if config[0]=='lt':", """    if config[0]=='lt_nn':
        key=(layer,x.shape[0],n,x.shape[1])
        state['mod'].lt_nn_run(state['lt_transposed'][layer],x,out,state['lt_workspace'],state['lt_nn_algorithms'][key],config[1])
        return out
    if config[0]=='lt':""")
s=s.replace("    state['lt_algorithms']={}","""    state['lt_algorithms']={}
    state['lt_nn_algorithms']={}
    state['lt_transposed']={layer:w.T.contiguous() for layer,w in weights.items() if layer not in ('router','qkv_gate')}""")
s=s.replace("                configs += [('lt',i) for i in range(codes.shape[0])]", """                configs += [('lt',i) for i in range(codes.shape[0])]
                nn_codes=state['mod'].lt_nn_options(state['lt_transposed'][layer],x,base_out)
                state['lt_nn_algorithms'][(layer,m,n,k)]=nn_codes
                configs += [('lt_nn',i) for i in range(nn_codes.shape[0])]""")
s=s.replace("if config[0]!='lt':", "if config[0] not in ('lt','lt_nn'):")
s=s.replace("scratch if selected[0]!='lt' else None", "scratch if selected[0] not in ('lt','lt_nn') else None")
s=s.replace("print(f'REDUCE_HIER ","print(f'FUSED ")
nn=builder.assignments((P/'cand_lt_layout.py').read_text())['LT_NN_SRC']
s+='\nFUSED_CONFIGS = '+repr(configs)+"\nLT_NN_SRC = r'''"+nn+"'''\n"
out=P/'cand_fused.py';compile(s,str(out),'exec');out.write_text(s);print('compact fused specializations',len(specs),'source bytes',len(s))
