"""Generate best5 from the immutable best source and the H200-proven NT2 body."""
from pathlib import Path
import ast,hashlib,re
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best.py'
OUT=BASE.with_name('cand_r3_best5.py')
BASE_SHA='812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2'

def literal(s,key):
    return ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets)))

def main():
    source=BASE.read_text();assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
    lab=(ROOT/'prof/lab/moe_v9_reuse_v2.py').read_text();native=literal(lab,'base_CUDA_SRC')
    # Kernel body is byte-for-byte the accepted lab snapshot. The C++ bindings
    # instantiate only NT2. Keep best's original NT1 extension entirely intact.
    native=native.replace('C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin)); ','').replace(' C10_CUDA_CHECK(cudaFuncSetAttribute(moe_fused<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));','')
    a=native.index('  if(nt==1){C10_CUDA_CHECK(cudaOccupancy');b=native.index('\n  return n;',a)
    native=native[:a]+'  TORCH_CHECK(nt==2,"NT2 extension requires NT2");\n  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, moe_fused<2>, NTHR, smem_bytes(nt,nstage,M)));'+native[b:]
    a=native.index('  if(nt==1){LAUNCH(1);}');b=native.index('\n#undef LAUNCH',a)
    native=native[:a]+'  TORCH_CHECK(nt==2,"NT2 extension requires NT2");\n  LAUNCH(2);'+native[b:]
    assert 'moe_fused<1>' not in native and 'moe_fused<4>' not in native
    source=source.replace('    nt = 1\n    nstage = 2\n','    nt = state.get("nt",1)\n    nstage = 2\n',1)
    source=source.replace('                        state["moe_bank"])','                        dict(state["moe_bank"],mod=state["moe_nt2_mod"],nt=2)\n                        if state.get("moe_nt_plan",{}).get(x.shape[0],1)==2 else state["moe_bank"])',1)
    # Every old tuner checks the one deadline; ordinary behavior stays unchanged
    # when time remains. Defaults are the already tested best choices.
    source=source.replace('    if _fa3_on_device(device):\n        try:\n            state["lt_mod"]', '    if _fa3_on_device(device) and _best5_left(10):\n        try:\n            state["lt_mod"]',1)
    source=source.replace('            w=state[name]; out_dtype=', '            if not _best5_left():\n                plan[(t,name)]="custom";continue\n            w=state[name]; out_dtype=',1)
    source=source.replace('            for variant in variants:\n','            for variant in variants:\n                if not _best5_left():break\n',1)
    source=source.replace('    for t in _TUNE_TOKENS:\n        stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)\n        for phase in ("b1","b2"):', '    for t in _TUNE_TOKENS:\n        if not _best5_left():\n            plan[t,"b1"]=False;plan[t,"b2"]=False;continue\n        stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)\n        for phase in ("b1","b2"):',1)
    source=source.replace('            scores=[]\n            for enabled in (False,True):','            scores=[]\n            for enabled in (False,True):\n                if not _best5_left():break\n',1)
    source=source.replace('    if not _attn_port_hopper(device):\n','    if not _attn_port_hopper(device) or not _best5_left(10):\n',1)
    source=source.replace('        for choice in _attn_port_candidates(length):\n','        for choice in _attn_port_candidates(length):\n            if not _best5_left():break\n',1)
    source=source.replace("        if ('layer',None) not in admitted:\n", "        if not admitted and not _best5_left():continue\n        if ('layer',None) not in admitted:\n",1)
    source=source.replace('                qkv,data = _attn_port_synthetic(batch,length,window,device)\n', '                if not _best5_left():continue\n                qkv,data = _attn_port_synthetic(batch,length,window,device)\n',1)
    source=source.replace('                for choice in admitted:\n','                for choice in admitted:\n                    if not _best5_left():break\n',1)
    source=source.replace('                us, choice=min(scores,key=lambda x:x[0])\n','                if not scores:continue\n                us, choice=min(scores,key=lambda x:x[0])\n',1)
    source=source.replace('    for t in _TUNE_TOKENS:\n        stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)\n        attn=', "    for t in _TUNE_TOKENS:\n        if not _best5_left():\n            plan[t]=('regular',state['pdl_plan'].get((t,'b1'),True));continue\n        stream=torch.randn((t,HIDDEN),device=device,dtype=torch.float32)\n        attn=",1)
    source=source.replace('        for enabled in (False,True):\n            fn=lambda enabled=enabled: _prefix_variant', '        for enabled in (False,True):\n            if not _best5_left():break\n            fn=lambda enabled=enabled: _prefix_variant',1)
    source=source.replace('    if not _fa3_on_device(device):\n        return\n    import hashlib\n    import statistics\n','    if not _fa3_on_device(device) or not _best5_left(10):\n        return\n    import hashlib\n    import statistics\n',1)
    source=source.replace('    for (t,name),(nn,ordinal) in _MEASURED_LT.items():\n','    for (t,name),(nn,ordinal) in _MEASURED_LT.items():\n        if not _best5_left():break\n',1)
    source=source.replace('            for repeat in range(4):\n','            for repeat in range(4):\n                if not _best5_left():break\n',1)
    source=source.replace('            ratio=statistics.median(ratios)\n','            if len(ratios)!=4:continue\n            ratio=statistics.median(ratios)\n',1)
    source=source.replace('    started=time.monotonic()\n    state=_prepare_weights(weights)\n','''    global _BEST5_DEADLINE
    started=time.monotonic();_BEST5_DEADLINE=started+240.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        native=pool.submit(_best5_nt2_module)
        state=_prepare_weights(weights)
        state['moe_nt2_mod']=native.result()
''',1)
    source=source.replace("    flush=torch.empty(256<<20,device=device,dtype=torch.uint8)\n    state['proj_plan']=", "    flush=torch.empty(256<<20,device=device,dtype=torch.uint8)\n    _best5_tune_moe(state,weights,flush,device)\n    phases['moe_nt']=time.monotonic()-started-sum(phases.values())\n    state['proj_plan']=",1)
    source=source.replace("f'pdl={tuple(state[\"pdl_plan\"][(t,p)] for p in (\"b1\",\"b2\"))} moe=v9'", "f'pdl={tuple(state[\"pdl_plan\"][(t,p)] for p in (\"b1\",\"b2\"))} moe=v9nt{state[\"moe_nt_plan\"].get(t,1)}'",1)
    source=source.replace("    return state\n\n\n# Projection shortlist", "    for m,row in state['moe_nt_rows'].items():print(f'BEST5_NT M={m} {row}',flush=True)\n    return state\n\n\n# Projection shortlist",1)
    source+='\n\nV9_NT2_CUDA_SRC = '+repr(native)+'\nV9_NT2_CPP_SRC = '+repr(literal(lab,'base_CPP_SRC'))+'\n\n'+(Path(__file__).parent/'tuner.py.inc').read_text()
    OUT.write_text(source)
    print(OUT.relative_to(ROOT),hashlib.sha256(OUT.read_bytes()).hexdigest())

if __name__=='__main__':main()
