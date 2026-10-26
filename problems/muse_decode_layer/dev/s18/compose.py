"""One S18 composition of pinned pending nominations on the frozen S17."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
sys.path.insert(0,str(H.parent/'s16'));import patches
BASE_SHA='fea717603233f240b3e5d2c99b5ffdd0edb19d4264161489da44cc192b7f55dc'
DEFAULT=['pool128','Vwait16','Vwait32','BLOCK','INDEX1','pdl_M','pdl_D','D8','D4']
# Only point gains whose published 95% intervals cross unity get leave-outs.
# Composition C's release/O128 pieces are already in S17, so C adds only pool128,
# whose S16 leave-out (37c2879f) and S15 pool-only (60bad72b) intervals cross unity.
UNCERTAIN=['Vwait32','BLOCK','D8','D4','pool128']
# Gate audit: M (metadata before wait) has the highest R of the B/M alternatives;
# B is M plus _gb/_gi L2 hints and measured lower on the same seeds (0f74021a).
PREPARE_TAIL='\ndef prepare(weights):\n    return _s16_prepare_delta(weights,_s16_origin_prepare(weights))\n'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()

def baseline():
 s=(P/'cand_r3_best7_s17_fea717603233.py').read_text()
 n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
 s=lzma.decompress(base64.b85decode(ast.literal_eval(n.args[0]))).decode();assert sha(s)==BASE_SHA;return s

def assign(source,name,value):
 nodes=patches.bindings(source)[name];assert len(nodes)==1
 n=nodes[0];lines=source.splitlines(True);lines[n.lineno-1:n.end_lineno]=[name+' = '+repr(value)+'\n'];return ''.join(lines)

def build(enabled=None):
 enabled=set(DEFAULT if enabled is None else enabled);assert enabled<=set(DEFAULT)
 source=baseline();raw=(H/'cells.json').read_bytes();manifest=json.load(open(H/'cells_manifest.json'));assert hashlib.sha256(raw).hexdigest()==manifest['sha256'];cells=json.loads(raw)
 # C's attention/large-M/NF release edits are already in S17 byte for byte.
 nodes=patches.bindings(source)
 for d in cells['C']:
  if d['name'] not in ('ATTN_PORT_CUDA_SRC','_attn_port_ext','NF_CUDA'):continue
  (node,)=nodes[d['name']];new=ast.parse(d['new']).body[0]
  assert ast.dump(node)==ast.dump(new),(d['name'],'C edit missing from S17')
 if 'pool128' in enabled:
  # library_pool128 handles S15n's range dispatcher and dependent VDSK chain.
  # patch() ends by rebinding prepare to the S16 origin chain; S17's chain
  # (MT, P10 pool, RT tails) already ends in the global _s16_prepare_delta.
  source=patches.patch(source,patches.registry()['layer_library_pool128'])
  assert source.endswith(PREPARE_TAIL);source=source[:-len(PREPARE_TAIL)]
 if 'pdl_M' in enabled:
  source=patches.apply_delta(source,cells['pdl_M'])
 if 'Vwait16' in enabled or 'Vwait32' in enabled:
  old_vq=ast.literal_eval(ast.parse(cells['Vwait'][0]['old']).body[0].value)
  new_vq=ast.literal_eval(ast.parse(cells['Vwait'][0]['new']).body[0].value)
  current=ast.literal_eval(patches.bindings(source)['VQ8_CUDA'][0].value)
  value=patches.merge(old_vq,new_vq,current)
  source=assign(source,'VQ8_CUDA',value)
  # Optional row leave-out preserves the other row's selected kernel.
  if ('Vwait16' in enabled)!=('Vwait32' in enabled):
   prefix='_s18_vwait_off_';off=prefix+'CUDA';source+='\n'+off+'='+repr(current)+'\n'+prefix+'MOD=None\n'
   module=patches.segment(source,patches.bindings(source)['_vq8_module'][0]);module=module.replace('_vq8_module',prefix+'module').replace('_VQ8_MOD',prefix+'MOD').replace('VQ8_CUDA',off).replace('muse_vq8_9a23ca9a98dd','s18_vqoff_'+sha(current)[:8]);source+='\n'+module
   driver=patches.segment(source,patches.bindings(source)['_vq8_attention'][0]);driver=driver.replace('def _vq8_attention(',f'def {prefix}attention(').replace('_vq8_module()',prefix+'module()');source+='\n'+driver
   kept=16 if 'Vwait16' in enabled else 32
   source+=f'''\n_s18_vq_selected=_vq8_attention
def _vq8_attention(qkv,data,setting):
    if data['config']['batch']=={kept}:return _s18_vq_selected(qkv,data,setting)
    return {prefix}attention(qkv,data,setting)
'''
  # Source-addressed module name prevents loading the incumbent binary.
  node=patches.bindings(source)['_vq8_module'][0];old=patches.segment(source,node);new=old.replace('muse_vq8_9a23ca9a98dd','muse_s18_vq_'+sha(value)[:12]);source=source.replace(old,new,1)
 names=['small_index' if 'INDEX1' in enabled else 'small_shard','small_pair','small_block' if 'BLOCK' in enabled else 'small_pair'];prefixes=[]
 emitted=set()
 for name in names:
  if name in emitted:
   prefixes.append(manifest['provenance'][name]['prefix']);continue
  emitted.add(name)
  code=cells[name];prefix=manifest['provenance'][name]['prefix'];prefixes.append(prefix)
  if 'pdl_M' in enabled:
   change=next(x for x in cells['pdl_M'] if x['name']=='SMALLM_VAR_CUDA');a=ast.literal_eval(ast.parse(change['old']).body[0].value);b=ast.literal_eval(ast.parse(change['new']).body[0].value)
   cuda_name=prefix.upper()+'_VAR_CUDA' if prefix.upper()+'_VAR_CUDA' in patches.bindings(code) else prefix.upper()+'_CUDA';node=patches.bindings(code)[cuda_name][0];v=ast.literal_eval(node.value);code=assign(code,cuda_name,patches.merge(a,b,v))
  source+='\n'+code
 source+='\n_s18_old_smallm=smallm_experts\ndef smallm_experts(data,state,flags,stages=2,grid=264):\n    m=data["hidden_states"].shape[0]\n'
 for p,ms in zip(prefixes,[(1,),(8,),(16,32)]):source+=f'    if m in {ms}:return {p}_experts(data,state,flags,stages,grid)\n'
 source+='    return _s18_old_smallm(data,state,flags,stages,grid)\n'
 # Prepare's FP32 expert gate runs smallm_experts right after the banks, before
 # _s16_prepare_delta: bind there too (as patches.patch does for epi cells).
 # Counters are keyed by flags word: SMALLM_FLAGS is {1: M=1 flags, 64: M>1 flags}.
 source+='\n_s18_old_banks=_s4_prepare_banks\ndef _s4_prepare_banks(state,weights):\n    _s18_old_banks(state,weights)\n'
 for p in dict.fromkeys(prefixes):source+=f'    {p}_bind(state["s4_nt1_bank"],SMALLM_FLAGS)\n'
 # Compile in prepare's parallel native pool; S17 prepare already takes 236 s of 300 on the H200.
 source+='\n_s16_native_builds.extend(['+','.join(p+'_module' for p in dict.fromkeys(prefixes))+'])\n'
 # The S16 builds are submitted first; give each its own worker so they never
 # queue the base pool's attention/NF compiles behind them (S17 judge prepare 198-255 s).
 pool='    with ThreadPoolExecutor(max_workers=15) as native_pool:\n';assert source.count(pool)==1
 source=source.replace(pool,'    with ThreadPoolExecutor(max_workers=15+len(_s16_native_builds)) as native_pool:\n')
 if 'pdl_D' in enabled:source=patches.apply_delta(source,[d for d in cells['pdl_D'] if d['name']!='prepare'])+'\n_s16_native_builds.append(gp_module)\n'
 if 'D8' in enabled or 'D4' in enabled:
  rows=ast.literal_eval(patches.bindings(source)['_R2_SELECTION'][0].value)
  if 'D8' in enabled:rows[16,1]=('dsm',1,8)
  if 'D4' in enabled:rows[32,1]=('dsm',1,4)
  source=assign(source,'_R2_SELECTION',rows)
 source+='\n_s18_prior_delta=_s16_prepare_delta\ndef _s16_prepare_delta(weights,state):\n    state=_s18_prior_delta(weights,state)\n'
 for p in dict.fromkeys(prefixes):source+=f'    {p}_bind(state["s4_nt1_bank"],SMALLM_FLAGS)\n'
 if 'pdl_D' in enabled:source+='    gp_module()\n    for key,plan in state["best7_plans"].items():plan["gp_key"]=key\n'
 if 'D8' in enabled or 'D4' in enabled:source+='    state["r2_rows"]=_r2_prepare(weights["qkv_gate"].device)\n'
 source+='    return state\n'
 compile(source,'<S18>','exec')
 return source,dict(base_sha256=BASE_SHA,sha256=sha(source),enabled=sorted(enabled),uncertain_leaveouts=UNCERTAIN,cells_sha256=manifest['sha256'])
if __name__=='__main__':
 s,m=build();(H/'generated/s18.py').write_text(s);(H/'selection.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
