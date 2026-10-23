"""Extend the frozen all-shape portfolio with shared-X tiles, multicast and DeepGEMM."""
import ast
import hashlib
import json
from pathlib import Path
import sys
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'dev'))
from lean_codegen import values_of,device_only
from build_portfolio_v2 import replacement
from build_portfolio_all import function

if '--native' in sys.argv:
    source=(P/'cand_portfolio_max.py').read_text()
    source=source.replace("state['deep_gemm_fn']=_dg_load()", "state['deep_gemm_fn']=_dg_load()\n    state['deep_gemm_cublas_fn']=_dg_cublas_load()")
    source=source.replace("return deep_gemm.bf16_gemm_nt", "deep_gemm.use_deterministic_algorithms(True)\n    return deep_gemm.bf16_gemm_nt")
    source=source.replace("('tile','wide','deep_gemm')", "('tile','wide','deep_gemm','deep_gemm_cublas')")
    source=source.replace("    else:state['deep_gemm_fn'](x,state[layer][:n],out)","    elif config[0]=='deep_gemm':state['deep_gemm_fn'](x,state[layer][:n],out)\n    else:state['deep_gemm_cublas_fn'](x,state[layer][:n],out)")
    source=source.replace("if layer!='router' and m>=16:yield ('deep_gemm',)","if layer!='router' and m>=16:\n        yield ('deep_gemm',)\n        yield ('deep_gemm_cublas',)")
    source+='\n\ndef _dg_cublas_load():\n    import deep_gemm\n    return deep_gemm.cublaslt_gemm_nt\n'
    output=P/'cand_portfolio_native.py';compile(source,str(output),'exec')
    with output.open('x') as f:f.write(source)
    print(json.dumps({'file':str(output),'sha256':hashlib.sha256(source.encode()).hexdigest()},indent=2))
    sys.exit(0)

source=(P/'cand_portfolio_all.py').read_text();values=values_of(source);raw=values['RAW_CUDA']
tile_text=(P/'cand_tile.py').read_text();tv=values_of(tile_text);tile=tv['CUDA_SRC']
a=tile.index('__device__ __forceinline__ void cp_async16');z=tile.index('template <int NT, int TR, int U, int NA, int ST, int LD>\nvoid tile_launch_t',a)
tile='namespace muse_tile { using muse_proj::ld_stream;using muse_proj::mma16816;\n'+tile[a:z]+'\n}\n'
tile_specs=[(nt,tr,2,na,st,0) for nt in (4,8,16) for tr in (1,2) for na in (1,2) for st in (3,4) if not(nt==16 and tr==2 and na==2)]
wg_text=(P/'cand_wg2.py').read_text();wv=values_of(wg_text);ns={**wv}
for name in ('_wgmma_fn','wg_source'):function(wg_text,name,ns)
wide_specs=wv['WG_INSTANCES']
wide=device_only(ns['wg_source'](wide_specs),'muse_wg').replace('muse_wg','muse_wide')
host=(P/'dev/max_host.cuh').read_text()
host=host.replace('MAX_TILE_CASES',' '.join('MAX_TILE('+','.join(map(str,c))+')' for c in tile_specs))
host=host.replace('MAX_WIDE_CASES',' '.join('MAX_WIDE('+','.join(map(str,c))+','+str(of).lower()+')' for c in wide_specs for of in (False,True)))
raw+='\n'+tile+'\n'+wide+'\n'+host
source=replacement(source,'RAW_CUDA',raw)
source=source.replace('_MOD=_AllModule(path)','_MOD=_MaxModule(path)')
source=source.replace('def _all_run(data,state,config):','def _base_run(data,state,config):')
source=source.replace('def _all_variants(state,layer,m,n,k,props):','def _base_variants(state,layer,m,n,k,props):')
source=source.replace("    total_trials=0;report=[]","    state['deep_gemm_fn']=_dg_load()\n    total_trials=0;report=[]")
source=source.replace('muse_proj_all_{tag}_','muse_proj_max_{tag}_')
source+='\n'+(P/'dev/portfolio_max_tail.py').read_text()
source+='\nTILE_SPECS = '+repr(tile_specs)+'\nWIDE_SPECS = '+repr(wide_specs)+'\n'
output=P/'cand_portfolio_max.py';compile(source,str(output),'exec')
with output.open('x') as f:f.write(source)
print(json.dumps({'file':str(output),'sha256':hashlib.sha256(source.encode()).hexdigest(),'additional_tile_kernels':len(tile_specs),'additional_wide_kernels':2*len(wide_specs)},indent=2))
