import importlib.util,subprocess,json
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
p=Path('problems/muse_prefill_layer/cand_attention_fp8.py')
s=importlib.util.spec_from_file_location('fp8_attention_tiles',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
sig=dict(q_ptr='*fp8e4nv',kc_ptr='*fp8e4nv',vc_ptr='*fp8e4nv',pt_ptr='*i32',seq_ptr='*i32',o_ptr='*bf16',ks_ptr='*fp32',vs_ptr='*fp32',scale='fp32',L='i32',max_pages='i32')
results=[]
for bq,bn,g,w,st in ((16,128,8,8,2),(16,128,8,16,2),(16,128,8,8,1),(8,128,8,8,2),(8,256,8,8,2),(16,256,8,8,1),(8,128,8,4,2)):
  const=dict(PAGE=1,WINDOW=0,BQ=bq,BN=bn,GROUP=g,PSCALE=448.,CONTIG=True)
  sig.update({key:'constexpr' for key in const})
  c=triton.compile(ASTSource(fn=m._attn_fp8_kernel,signature=sig,constexprs=const),target=GPUTarget('cuda',90,32),options=dict(num_warps=w,num_stages=st))
  file=Path(f'problems/muse_prefill_layer/dev/attention_fp8_tile_{bq}_{bn}_{g}_{w}_{st}_sm90.cubin');file.write_bytes(c.asm['cubin'])
  res=subprocess.run(['cuobjdump','--dump-resource-usage',str(file)],capture_output=True,text=True,check=True).stdout
  row=dict(bq=bq,bn=bn,group=g,warps=w,stages=st,shared=c.metadata.shared,resource=res)
  print(row,flush=True);results.append(row)
Path('problems/muse_prefill_layer/dev/attention_fp8_compile_tiles.json').write_text(json.dumps(results,indent=2))
