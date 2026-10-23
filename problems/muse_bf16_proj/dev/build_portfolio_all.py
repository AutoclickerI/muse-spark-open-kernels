"""Build one self-contained, broad H200-tuned portfolio; never change queued sources."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'dev'))
import build_portfolio_v2 as b
from lean_codegen import values_of,device_only
from receipt_digest import digest


def function(text,name,scope):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(ast.get_source_segment(text,node),scope)


def safe_tuner(source, *, extended=False, native=False):
    """Refresh inherited tuning code without changing a frozen input source.

    Max/native builders read archived candidates. Their generated copies must
    still get the current rejection handling and prepare-time trial deadline.
    """
    template=(P/'dev/portfolio_all_tail.py').read_text()
    updates={n.name:ast.get_source_segment(template,n) for n in ast.parse(template).body
             if isinstance(n,ast.FunctionDef) and n.name in ('prepare','_all_variants')}
    if extended:
        updates['_base_variants']=updates.pop('_all_variants').replace('def _all_variants(', 'def _base_variants(',1)
        setup=[]
        for key,loader in [('deep_gemm_fn','_dg_load')]+([('deep_gemm_cublas_fn','_dg_cublas_load')] if native else []):
            setup.append(f"    try:state[{key!r}]={loader}()\n"
                         f"    except Exception as exc:\n"
                         f"        state[{key!r}]=None\n"
                         f"        print('ALL_REJECT loader {key}:',str(exc)[:180],flush=True)\n")
        updates['prepare']=updates['prepare'].replace(
            '    total_trials=0;report=[]',
            ''.join(setup)+'    total_trials=0;report=[]')
    lines=source.splitlines(keepends=True)
    for n in reversed(ast.parse(source).body):
        if isinstance(n,ast.FunctionDef) and n.name in updates:
            lines[n.lineno-1:n.end_lineno]=[updates[n.name]+'\n']
    return ''.join(lines)


def build(output):
    original=(P/'cand_portfolio.py').read_text();ov=values_of(original)
    base=(P/'cand_fused_select.py').read_text();bv=values_of(base)
    # Include all original tile/stage combinations and all producer schedules.
    tiles=list(ov['HOP_CONFIGS'])+[(128,128,3)]
    typed=[]
    for bt,bk,st in tiles:
        for th in (128,160,256):
            for of in (False,True):typed.append(((bt,bk,st,0,1,th),of))
    hierarchical=[]
    for bt,bk,st in ((64,64,4),(128,64,3),(256,64,2)):
        for th in (128,160,256):
            c=(bt,bk,st,3,32,th);typed.append((c,True));hierarchical.append(c)
    pair,specs=b.fused_source(typed);pair=device_only(pair,'muse_pair')
    ns={**ov};function(original,'_wgmma_fn',ns);function(original,'_hop_source',ns)
    old=device_only(ns['_hop_source'](),'muse_wg').replace('muse_wg','muse_legacy')
    # Every known small-M dispatch configuration remains eligible, plus independent
    # warp/cluster partitioning of the same arithmetic kernels.
    skin={};instances=set(bv['INSTANCES'])
    for name in ('submission.py','cand_tuned.py','cand_tuned2.py','cand_hybrid.py','cand_skinny.py','cand_gemv.py','cand_gemv2.py','cand_gemv3.py'):
        vals=values_of((P/name).read_text())
        for layer in ('qkv_gate','o','pooler','unpooler','router'):
            for m in (1,8,16,32,128,256):
                nt=1 if m<=8 else 2 if m<=16 else 4 if m<=32 else 8
                bucket=m if m<=32 else 64
                cfg=vals.get('TUNED_CONFIGS',{}).get((layer,bucket),vals.get('CONFIGS',{}).get((layer,nt)))
                if cfg is None:continue
                tr,u,wn,wk,ks,xp=cfg
                c=(nt,tr,u,wn,wk,ks,xp,0)
                skin.setdefault((layer,m),set()).add(c)
                instances.add((nt,tr,u,2 if layer=='router' else 1,xp,0))
    # Broader one/two-wave small-M coverage, without recompiling each WN/WK/KS.
    for layer in ('qkv_gate','o','pooler','unpooler','router'):
        for m in (1,8,16,32,128,256):
            nt=1 if m<=8 else 2 if m<=16 else 4 if m<=32 else 8
            for tr,u,xp in ((1,4,1),(2,2,1),(1,2,0)):
                if nt==8 and (tr!=1 or xp!=0):continue
                for wn,wk in ((1,8),(2,4),(4,2),(8,1),(1,4),(2,2)):
                    for ks in (1,2,4,8,16):
                        skin.setdefault((layer,m),set()).add((nt,tr,u,wn,wk,ks,xp,0))
                instances.add((nt,tr,u,2 if layer=='router' else 1,xp,0))
    skin={k:sorted(v) for k,v in skin.items()}
    # The arithmetic body is the measured skinny kernel, with only its ABI stripped.
    body=ov['CUDA_SRC'];end=re.search(r'template <int NT, int TR, int U, int NA, int XP, int LD>\nvoid launch_t',body).start()
    body=body[:end]+'\n} // namespace muse_proj\n'
    body=body.replace('#include <torch/extension.h>','').replace('#include <ATen/cuda/CUDAContext.h>','').replace('namespace cg = cooperative_groups;','')
    raw=(P/'dev/lean_host.cuh').read_text()
    raw=raw.replace('LEAN_DEVICE_SKINNY',body+old).replace('LEAN_DEVICE_PAIR',pair)
    raw=raw.replace('LEAN_SKIN_CASES',' '.join('LSKIN('+','.join(map(str,c))+')' for c in sorted(instances)))
    raw=raw.replace('LEAN_PAIR_CASES',' '.join(f'LPAIR({bt},{bk},{st},{th},{str(of).lower()},{str(hh).lower()})' for bt,bk,st,th,of,hh in specs))
    raw=raw.replace('constexpr int BT=NT*8;int BR=wn*TR*16;','constexpr int BT=NT*8;int BR=wn*TR*16;\n  if(ks>8){auto skinny_kern=muse_proj::skinny_kernel<NT,TR,U,NA,XP,LD>;LEAN_CUDA(cudaFuncSetAttribute(skinny_kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));}')
    raw+='\n'+(P/'dev/lean_old_host.cuh').read_text().replace('LEAN_OLD_CASES',' '.join(f'LOLD({bt},{bk},{st},{str(of).lower()})' for bt,bk,st in ov['HOP_CONFIGS'] for of in (False,True)))
    raw+='\n'+(P/'dev/all_lt.cuh').read_text()
    misc=[]
    for i,name in enumerate(('cand_gemv.py','cand_gemv2.py','cand_gemv3.py','cand_gemv_static.py')):
        src=values_of((P/name).read_text())['CUDA_SRC']
        a=src.index('__device__ __forceinline__ float dot8_bf16')
        z=src.index('void gemv_impl(',a)
        misc.append(f'namespace all_gemv{i} {{\n'+src[a:z]+'\n}\n')
    tc=values_of((P/'cand_tc.py').read_text())['CUDA_SRC']
    a=tc.index('// M>=8');z=tc.index('template <int BT, int KT>\nvoid launch_tc',a)
    misc.append('namespace all_tc { using muse_proj::mma16816;\n'+tc[a:z]+'\n}\n')
    scalar=values_of((P/'cand_m16.py').read_text())['CUDA_SRC']
    a=scalar.index('__device__ __forceinline__ float dot8_m16');z=scalar.index('void m16(torch',a)
    misc.append('namespace all_scalar {\n'+scalar[a:z]+'\n}\n')
    raw+='\n'+''.join(misc)+(P/'dev/all_misc_host.cuh').read_text()
    # Reuse the checked module/graph ownership and measured incumbent unchanged.
    keep=[]
    names={'_prepare_small','_nt','skinny_gemm','_small','_hopper','_TuneGraph','_graph','_LeanModule','_module'}
    assignments={'INSTANCES','CONFIGS','TUNED_CONFIGS','SKINNY_MAX_M','_MOD'}
    for node in ast.parse(base).body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):keep.append(ast.get_source_segment(base,node))
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:keep.append(ast.get_source_segment(base,node))
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in node.targets):keep.append(ast.get_source_segment(base,node))
        elif isinstance(node,ast.ClassDef) and node.name in ('Weights','Config','Inputs'):keep.append(ast.get_source_segment(base,node))
    source='# okc: problem=muse_bf16_proj\n"""All-shape synthetic cold-L2 portfolio. No benchmark-input state is cached."""\n'+'\n\n'.join(keep)
    source=source.replace('_MOD=_LeanModule(path)','_MOD=_AllModule(path)')
    source=source.replace("'-Xptxas=-v'", "'-Xptxas=-v','--split-compile=16','-Xptxas=--split-compile=16'")
    source=source.replace('muse_proj_lean_{tag}_','muse_proj_all_{tag}_')
    evidence={};winners={}
    for receipt in sorted((P.parents[1]/'runs/muse_bf16_proj').glob('*.json')):
        d=digest(receipt)
        if d['source_sha256']==b.FROZEN and d['passed'] and not d['flags']:
            winners={(k.split(':')[0],int(k.split(':')[1])):tuple(v) if v else None for k,v in d['choices'].items()}
            evidence={'job':d['job'],'source_sha256':d['source_sha256']}
    source+='\n\n'+(P/'dev/portfolio_all_tail.py').read_text()
    source+='\nSKIN_VARIANTS = '+repr(skin)+'\nPAIR_TILES = '+repr(tiles)+'\nOLD_TILES = '+repr(ov['HOP_CONFIGS'])+'\nHIER_CONFIGS = '+repr(hierarchical)+'\n'
    source+='H200_WINNERS = '+repr(winners)+'\nH200_EVIDENCE = '+repr(evidence)+"\nRAW_CUDA = r'''"+raw+"'''\n"
    compile(source,str(output),'exec')
    with output.open('x') as f:f.write(source)
    print(json.dumps({'file':str(output),'sha256':hashlib.sha256(source.encode()).hexdigest(),
                      'pair_specializations':len(specs),'old_specializations':len(ov['HOP_CONFIGS'])*2,
                      'skinny_specializations':len(instances),'h200_evidence':evidence},indent=2))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=P/'cand_portfolio_all.py');args=ap.parse_args();build(args.output)
