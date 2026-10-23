"""Build a short portfolio only from the completed, exact-source H200 portfolio receipt.
Old H200 winners compete with the new fused paths and lossless NN weight layout.
The generated submission is self-contained; this tool is never imported by it.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]
sys.path.insert(0,str(P/'dev'))
from receipt_digest import digest
FROZEN='57086918cce13065f97757c2471b7b79072e2bc45e8c589899723bf1d70d748e'
LAYERS=('router','qkv_gate','pooler','unpooler','o')


def assignments(text):
    return {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(text).body
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
            and isinstance(node.value,(ast.Constant,ast.Tuple,ast.List,ast.Dict))}


def replacement(text,name,value):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    lines=text.splitlines(keepends=True);a=sum(map(len,lines[:node.lineno-1]));b=sum(map(len,lines[:node.end_lineno]))
    return text[:a]+f'{name} = {value!r}\n'+text[b:]


def fused_source(configs):
    text=(P/'cand_reduce_hier.py').read_text();values=assignments(text)
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_wgmma_fn')
    ns={};exec(ast.get_source_segment(text,node),ns)
    bts=sorted({c[0] for c,of in configs})
    specs=sorted({(c[0],c[1],c[2],c[5],bool(of),c[3]==3) for c,of in configs})
    fns=''.join(ns['_wgmma_fn'](n) for n in bts)
    mma=' else '.join(f'if constexpr(BT=={n}) wgmma_m64n{n}(acc,da+2*(kk%4),db+2*(kk%4),1);' for n in bts)
    cases=' '.join(f'HCASE({bt},{bk},{st},{th},{str(of).lower()},{str(hier).lower()})' for bt,bk,st,th,of,hier in specs)
    kernel=values['HOP_KERNEL']
    kernel=kernel.replace('bool OUTF32, int THREADS>', 'bool OUTF32, int THREADS, bool HIER>')
    kernel=kernel.replace('if (mode == 3) {','if constexpr (HIER) {')
    kernel=kernel.replace('template <int BT, int BK, int ST, int THREADS>', 'template <int BT, int BK, int ST, int THREADS, bool OUTF32, bool HIER>')
    kernel=kernel.replace('auto kern=Out.scalar_type()==at::kFloat ? hop_kernel<BT,BK,ST,true,THREADS> : hop_kernel<BT,BK,ST,false,THREADS>;',
                          'auto kern=hop_kernel<BT,BK,ST,OUTF32,THREADS,HIER>;')
    kernel=kernel.replace('#define HCASE(BT,BK,ST,TH) if(bt==BT && bk==BK && st==ST && threads==TH) { hop_launch<BT,BK,ST,TH>(W,X,Out,Work,Counts,mode,split); return; }',
                          '#define HCASE(BT,BK,ST,TH,OF,HH) if(bt==BT && bk==BK && st==ST && threads==TH && (Out.scalar_type()==at::kFloat)==OF && (mode==3)==HH) { hop_launch<BT,BK,ST,TH,OF,HH>(W,X,Out,Work,Counts,mode,split); return; }')
    source=values['HOP_HELPERS'].replace('WGMMA_FNS',fns)+kernel.replace('HOP_MMA',mma).replace('HOP_MAP',values['HOP_MAP']).replace('HOP_CASES',cases)
    source=source.replace('muse_wg','muse_pair').replace('void hopper(', 'void hopper_pair(').replace('HCASE','PAIR_CASE')
    return source,specs


def local_fused_choices():
    log=(P/'dev/reduce_hier_screen.log').read_text()
    out={}
    for line in log.splitlines():
        match=re.match(r'REDUCE_HIER (\w+) m=(\d+).*?top=(.*)',line)
        if match and match[1] in ('router','qkv_gate'):
            configs=[]
            for us,config in ast.literal_eval(match[3]):
                if tuple(config) not in configs: configs.append(tuple(config))
                if len(configs)==3:break
            out[(match[1],int(match[2]))]=configs
    return out


def assemble(winners,evidence):
    """Assemble source independently so all dispatch branches can be checked locally.

    Only build() turns actual H200 evidence into a candidate. Local assembly checks
    supply explicit control configurations, never a fabricated judge receipt.
    """
    original=(P/'cand_portfolio.py').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=FROZEN: raise RuntimeError('queued source changed')
    old_specs=sorted({c[:3] for c in winners.values() if c and c[0]!='lt'})
    new=local_fused_choices()
    for layer in ('router','qkv_gate'):
        for m in (128,256):
            cfgs=new.setdefault((layer,m),[])
            if layer=='router':
                # New method, not a previously measured loser: fill a full H200 wave
                # with 32 K partitions, then reduce across the last 16-CTA cluster.
                hierarchical=(128,64,3,3,32,160) if m==128 else (256,64,2,3,32,160)
                if hierarchical not in cfgs:cfgs.append(hierarchical)
            old=winners[(layer,m)]
            if old and old[0]!='lt':
                config=tuple(old)+(256,)
                bt,bk,st,*_=config
                if max(st*(64*(2 if layer=='router' else 1)+bt)*bk*2+st*16,bt*68*4)<=227*1024:
                    if config not in cfgs:cfgs.append(config)
            if not cfgs: raise RuntimeError(f'no validated fused configurations for {layer} {m}')
    pair_src,pair_specs=fused_source([(c,layer=='router') for (layer,m),row in new.items() for c in row])
    nn_src=assignments((P/'cand_lt_layout.py').read_text())['LT_NN_SRC']
    source=replacement(original,'HOP_CONFIGS',tuple(old_specs))
    source=source.replace(' + LT_SRC',' + LT_SRC + PAIR_SRC + LT_NN_SRC')
    source=source.replace('torch::Tensor lt_options(torch::Tensor W, torch::Tensor X, torch::Tensor Out);','''void hopper_pair(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Counts,
                 int64_t bt,int64_t bk,int64_t st,int64_t mode,int64_t split,int64_t threads);
torch::Tensor lt_nn_options(torch::Tensor W,torch::Tensor X,torch::Tensor Out);
void lt_nn_run(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Algos,int64_t index);
torch::Tensor lt_options(torch::Tensor W, torch::Tensor X, torch::Tensor Out);''')
    source=source.replace('functions=["skinny", "hopper", "lt_options", "lt_run"]','functions=["skinny", "hopper", "lt_options", "lt_run", "hopper_pair", "lt_nn_options", "lt_nn_run"]')
    a=source.index('def prepare(weights: Weights):');b=source.index('\nLT_SRC = ',a)
    new_prepare=Path(__file__).with_name('portfolio_v2_tail.py').read_text()
    source=source[:a]+new_prepare+source[b:]
    source+='\nH200_WINNERS = '+repr(winners)+'\nFUSED_CONFIGS = '+repr(new)+'\n'
    assert "'''" not in pair_src and "'''" not in nn_src
    source+="PAIR_SRC = r'''"+pair_src+"'''\nLT_NN_SRC = r'''"+nn_src+"'''\n"
    source+='H200_EVIDENCE = '+repr(evidence)+'\n'
    from lean_codegen import convert
    source,lean_specs=convert(source)
    return source,lean_specs


def build(receipt,output):
    output=output.resolve()
    evidence=digest(receipt)
    if evidence['source_sha256']!=FROZEN: raise RuntimeError('receipt does not match frozen portfolio source')
    if not evidence['passed'] or evidence['flags']: raise RuntimeError('portfolio is not a clean passing H200 receipt')
    expected={f'{l}:{m}' for l in LAYERS for m in (128,256)}
    if set(evidence['choices'])!=expected: raise RuntimeError(f'missing prepare choices: {expected-set(evidence["choices"])}')
    winners={(key.split(':')[0],int(key.split(':')[1])):tuple(value) if value else None for key,value in evidence['choices'].items()}
    source,lean_specs=assemble(winners,{'job':evidence['job'],'source_sha256':FROZEN})
    compile(source,str(output),'exec')
    # A generated file can already be queued by the orchestrator. Never replace it.
    with output.open('x') as stream:stream.write(source)
    metadata={'receipt':evidence['receipt'],'job':evidence['job'],'old_hopper_specializations':lean_specs['old_specializations'],
              'new_fused_specializations':lean_specs['new_fused_specializations'],'winners':{f'{l}:{m}':c for (l,m),c in winners.items()},
              'source':str(output.relative_to(ROOT)),'sha256':hashlib.sha256(source.encode()).hexdigest()}
    (P/'dev/portfolio_v2_build_local.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('receipt',type=Path);ap.add_argument('--output',type=Path,default=P/'cand_portfolio_v2.py');args=ap.parse_args();build(args.receipt,args.output)
