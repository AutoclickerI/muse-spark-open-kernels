"""Ship just the two nominated families; retain exact measured kernel bodies."""
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;o=H/'out';cu=(o/'lee_hybrid.cu').read_text()
a=cu.index('__device__ __forceinline__ void mb_wait(');z=cu.index('// 2^x on',a);mb=cu[a:z]
a=cu.index('__device__ __forceinline__ void wg_n16_qk(');z=cu.index('struct F {',a);dec=cu[a:z]
prefix=cu[:cu.index('template <bool PS1, bool FUSE, bool MRG, int X>')]
host=cu[cu.index('// Owned host ABI'):];start=host.index('  A a{};');end=host.index('  if(kind==1 || kind==-1){',start);init=host[start:end]
# Close the compile-time local/native NL choice already included in init.
host='''
void lee_attention(torch::Tensor qkv,torch::Tensor loc,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,
 torch::Tensor wo,torch::Tensor wml,torch::Tensor cnt,torch::Tensor out,torch::Tensor invf,torch::Tensor empty,
 int64_t kind,int64_t window,int64_t ns,int64_t x,double theta,double qk_factor,double sm_scale,double kv_scale,int64_t nmgcap){
 auto st=at::cuda::getCurrentCUDAStream();const int B=seq.size(0),L=qkv.size(1)/B;
 TORCH_CHECK(B==32 && L==1 && window && kind==1 && x==64,"selected decode32 sliding ABI");
'''+init+'''
 f.nmg=2;const int smd=nl*(DRS*16384+32768+4096+2048+2048)+256+nl*1024+1024;
 adf<nl,64>(B*ns/nl,smd,st,a,f);
 const auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
}
'''
a0=cu.index('// RoPE angle pos');z0=cu.index('template <bool CL, bool MU>',a0);rope=cu[a0:z0]
selected=prefix+mb+rope+dec+host;(o/'lee_selected.cu').write_text(selected)
s=(o/'lee_hybrid.py').read_text();tree=ast.parse(s);run=(H.parent/'public_ports/lee_runtime.py.inc').read_text();run=run[:run.index('def attention(')].replace("VCFG={(1,True):(-1,128,8,8),(1,False):(-2,304,128,68),(16,True):(-2,128,8,32836),(32,True):(-2,256,8,32836),(16,False):(-2,256,8,324),(32,False):(-2,256,8,324)}\n",'').replace("name='p5b_lee_'","name='p5b_hybrid_'")
run+='''
def attention_pair(pair,data,arm='F'):
 p,s=pair;c=data['config'];B,L=c['batch'],c['q_len'];assert B==32 and L==1 and c['window'] and arm=='F'
 dev=p.device;setup(dev);nl=1 if _LOCAL else 2;sms=torch.cuda.get_device_properties(dev).multi_processor_count
 span=min(data['page_table'].shape[1],c['window']+L);ns=max(nl,min((span+63)//64,nl*(sms//B)));ns-=ns%nl
 key=(1,B,L,ns)
 if key not in _BUF:_BUF[key]=(torch.empty(B*ns*16*64,device=dev),torch.empty(B*ns*16*2,device=dev),torch.zeros(2*B,device=dev,dtype=torch.int32))
 wo,wml,cnt=_BUF[key];out=torch.empty((B,L,1024),device=dev,dtype=torch.bfloat16)
 _MOD.lee_attention(p,data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],wo,wml,cnt,out,_INV[str(dev),float(c['rope_theta'])],_EMPTY,1,c['window'],ns,64,c['rope_theta'],c['qk_scale_factor'],c['sm_scale'],c['kv_scale'],2)
 return out
'''
(o/'lee_selected.py').write_text('CUDA='+repr(selected)+'\n'+run)
# All selected Triton kernel/helper ASTs are identical; remove full-only Gluon
# and host routing, keeping the measured sliding8 schedule unchanged.
s=(o/'operator_direct.py').read_text();t=ast.parse(s);lines=s.splitlines(True)
names={'_combine_ml','_combine','_emit_pre','_emit','_partial_load','_fused_attn','_counter','_fused','_cos_sin','setup','attention_pair'};parts=[]
for n in t.body:
 if isinstance(n,ast.FunctionDef) and n.name in names:
  first=min([n.lineno]+[d.lineno for d in n.decorator_list]);body=''.join(lines[first-1:n.end_lineno])
  if n.name=='_fused':
   a=body.index('    if (fc.get("gluon_v")');z=body.index('    group =',a);body=body[:a]+body[z:]
  parts.append(body)
header='''# MIT6d9b7765, Copyright2026bigboateng; full notice in shipped source.
# Changes: direct FP32 prologue; selected sliding8 family only.
import torch,triton
import triton.language as tl
RMS_EPS=1e-5
MERGE_ELEMS=tl.constexpr(16384)
_ROPE={};_COUNTERS={};ROPE_POSITIONS=1<<21
_DIRECT_PS=0;_DIRECT_SS=0;_DIRECT_WIDTH=2304
'''
fc=next(n for n in t.body if isinstance(n,ast.Assign) and any(getattr(x,'id',None)=='FUSED_CFG' for x in n.targets));cfg=next(v for k,v in zip(fc.value.keys,fc.value.values) if ast.literal_eval(k)==('sliding',8))
header+='FUSED_CFG={(\"sliding\",8):'+ast.unparse(cfg)+'}\n'
op=header+'\n\n'.join(parts)+'\n';(o/'operator_selected.py').write_text(op)
old_ast={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)};new_ast={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(op).body if isinstance(n,ast.FunctionDef)}
unchanged=sorted(names-{'_fused','attention_pair','setup'});assert all(old_ast[n]==new_ast[n] for n in unchanged)
assert dec in selected and dec in cu
proof=dict(selected_lee_cuda_sha256=hashlib.sha256(selected.encode()).hexdigest(),selected_operator_sha256=hashlib.sha256(op.encode()).hexdigest(),lee_attnd_body_identical=True,operator_kernel_AST_identical=unchanged,sizes=dict(lee_full=len(cu),lee_selected=len(selected),operator_full=len(s),operator_selected=len(op)))
(H/'pruning.json').write_text(json.dumps(proof,indent=2)+'\n');print(proof)
