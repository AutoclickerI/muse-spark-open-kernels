"""Freeze all useful expert layouts and per-projection tuning into one board source."""
import ast,re
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def function(source,name):
 tree=ast.parse(source);n=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 first=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
 return '\n'.join(source.splitlines()[first-1:n.end_lineno])+'\n\n'
def assignment(source,name):
 return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
base=(P/'cand_prefill_portfolio_v2.py').read_text()
# Discard old tuning and prepare definitions; keep the actual expert/attention kernels.
t=ast.parse(base)
drop={'_tune_portfolio','_tune_attention','prepare'}
ranges=[]
for n in t.body:
 if isinstance(n,ast.FunctionDef) and n.name in drop:ranges.append((n.lineno,n.end_lineno))
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_prepare_experts' for x in n.targets):ranges.append((n.lineno,n.end_lineno))
lines=base.splitlines()
for a,b in sorted(ranges,reverse=True):del lines[a-1:b]
base='\n'.join(lines)+'\n'
base=base.replace('Prefill portfolio: one canonical bank, BN64/128/256/DG expert selection and FP8 attention.',
 'Score portfolio: multiple lossless NVFP4 layouts and per-shape cold-graph stage selection.')
base=base.replace('def _experts_selected(', 'def _experts_canonical(')
# Kernel call remains the new final dispatch. Other calls in removed tuner do not survive.
base=base.replace('moe = _experts_canonical(', 'moe = _experts_selected(')
base=base.replace('qkv = torch.mm(x, st["qkv"].t())','qkv = _score_dense(x, st, "qkv")')
base=base.replace('a = torch.mm(o_in, st["o"].t())','a = _score_dense(o_in, st, "o")')
base=base.replace('logits = torch.mm(h, st["router_cat"].t(), out_dtype=torch.float32)','logits = _score_dense(h, st, "router_cat")')
base=base.replace('pooled = torch.mm(h, st["pooler"].t())','pooled = _score_dense(h, st, "pooler")')
base=base.replace('f = torch.mm(moe, st["unpooler"].t())','f = _score_dense(moe, st, "unpooler")')
raw=(P/'cand_v11.py').read_text()
# Shared Python gather and repacking for the original, shared/shared and bitplane banks.
extra=''
for name in ('_blob_index','cta_row_order','repack','tile_offsets_kernel','gather_tiled_kernel','combine_tiled_kernel','experts_tiled','_experts_dequant','_swiglu_kernel'):
 # Existing canonical frontend has its own incompatible tiled gather; prefix the raw family.
 src=function(raw,name)
 extra+=src
names=('_blob_index','cta_row_order','repack','tile_offsets_kernel','gather_tiled_kernel','combine_tiled_kernel','experts_tiled','_experts_dequant','_swiglu_kernel')
for n in sorted(names,key=len,reverse=True):extra=re.sub(r'\b'+n+r'\b','_raw_'+n.lstrip('_'),extra)
# BF16 effective postnorm, as stock and v11's serving reader.
extra=extra.replace("st['post_expert']", "st['post_expert_bf16']")
extra=extra.replace('st["post_expert"]','st["post_expert_bf16"]')
for name in ('_marlin_bank','_experts_marlin'):
 extra+=function(raw,name)
# Bit-plane encode is lossless and uses the same raw blob permutation.
fast=(P/'cand_v17.py').read_text()
for name in ('_encode_blob_fast','_encode_sf_fast','repack_fast'):
 extra+=function(fast,name).replace('b,s = repack(', 'b,s = _raw_repack(')
# Preserve the literal v11 DG path as well as tuned raw-dequant/SwiGLU paths.
extra+=function((P/'cand_v10.py').read_text(),'_swiglu_routes_kernel')
# Humming indexed (v6), indexed TMA (v14), and masked (v8) share one Humming bank.
hum=(P/'cand_v14.py').read_text()
imports=hum[hum.index('import json\nimport os\nfrom typing'):hum.index('class _Experts:')]
# The Humming import disables its process-spawning helper builds, exactly as checked v14.
extra+=imports
cls=function(hum,'_Experts')
cls=cls.replace('def __init__(self, weights, start: int, count: int):','def __init__(self, weights, start: int, count: int, tma=False):\n        self.tma=tma')
cls=cls.replace('        self.tma=tma','        self.tma=tma\n        self.gemm_type=GemmType.INDEXED')
cls=cls.replace('config = get_heuristics_config(', 'config = dict(get_heuristics_config(')
cls=cls.replace('gemm_type=GemmType.INDEXED)', 'gemm_type=self.gemm_type))')
cls=cls.replace('self.layer.humming_metas[name].to_str(), COMPUTE_CONFIG,', 'self.layer.humming_metas[name].to_str(), json.dumps({\"use_f16_accum\":False,\"gemm_type\":self.gemm_type.value}),')
cls=cls.replace('if torch.cuda.get_device_capability()[0] == 9:', 'if self.tma and torch.cuda.get_device_capability()[0] == 9:')
extra+=cls+function(hum,'_experts_humming')
masked=(P/'cand_v8.py').read_text()
extra+=function(masked,'_fixed_offsets_kernel')+function(masked,'_experts_humming_masked')
att=(P/'dev/attention_dense.py').read_text()
att=att.replace('def _attention_contiguous(data, q8, st):','def _attention_contiguous(data, q8, st, unpacked=False):')
att=att.replace('window_size=(W-1,0) if W else (-1,-1), **descale)', 'window_size=(W-1,0) if W else (-1,-1), pack_gqa=False if unpacked else None, num_splits=1 if unpacked else 0, **descale)')
att=att.replace('return _attention(packed, q8, st)','return _attention_fa3(packed, q8, st)')
extra+=att
# Compile one extension rather than recompiling Torch headers for each contender.
cuda=assignment(base,'GROUPED_SRC')
for v in (11,15,17):
 src=assignment((P/f'cand_v{v}.py').read_text(),'CUDA_SRC')
 src=re.sub(r'\bmoe\b',f'raw{v}',src)
 src=re.sub(r'\bmoe_gemm\b',f'raw{v}_gemm',src)
 cuda+='\n'+src
lt=assignment(Path('problems/muse_bf16_proj/cand_portfolio.py').read_text(),'LT_SRC')
lt=lt.replace('CUDA_R_16BF,n,m,n','(y.scalar_type()==at::kFloat ? CUDA_R_32F : CUDA_R_16BF),n,m,n')
# Spend a 256MiB Lt workspace: filter only the external library's valid heuristics.
lt=lt.replace('32*1024*1024','256*1024*1024')
cuda+='\n'+lt
# Replace the one literal assignment without AST reformatting Triton source.
node=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='GROUPED_SRC' for x in n.targets))
lines=base.splitlines();lines[node.lineno-1:node.end_lineno]=['GROUPED_SRC = '+repr(cuda)];base='\n'.join(lines)+'\n'
# Replace loader definition with the full exported interface.
old=function(base,'_grouped_module').rstrip()
loader=(P/'dev/score_loader.py').read_text().rstrip()
assert old in base
base=base.replace(old,loader,1)
base+='\n'+extra+'\n'+(P/'dev/score_portfolio_tail.py').read_text()
# Raw DG literal shares raw scales but applies the stock effective BF16 norm.
assert 'def _experts_selected(' in base
out=P/'cand_score_portfolio.py';out.write_text(base)
ast.parse(base)
print(out,len(base))
