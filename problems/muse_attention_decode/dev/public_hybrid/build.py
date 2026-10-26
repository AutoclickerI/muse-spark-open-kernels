"""Hybrids from reviewed, notice-preserved public kernels; no donor driver."""
import hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'public_ports';o=H/'out';o.mkdir(exist_ok=True)
# S19's direct partial consumption + OP both-head block diagonal/last-arriver.
op=(P/'operatorstack.py.inc').read_text();a=op.index('@triton.jit\ndef _fused_attn');z=op.index('@gluon.jit',a);part=op[a:z]
part=part.replace('QK8: tl.constexpr, GPRE: tl.constexpr):','QK8: tl.constexpr, GPRE: tl.constexpr, PS:tl.constexpr,PSS:tl.constexpr,PWIDTH:tl.constexpr):')
part=part.replace('tl.load(qptr, mask=qmask, other=0.0)','_partial_load(qptr,qmask,PS,PSS)').replace('tl.load(qptr + 1, mask=qmask, other=0.0)','_partial_load(qptr+1,qmask,PS,PSS)')
part=part.replace('tl.load(kptr, mask=jm[:, None], other=0.0)','_partial_load(kptr,jm[:,None],PS,PSS)').replace('tl.load(kptr + 1, mask=jm[:, None], other=0.0)','_partial_load(kptr+1,jm[:,None],PS,PSS)')
old='''        vraw = tl.load(qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride + (HQ + HKV + hblk * HPB) * D + dcol[None, :],
                       mask=jm[:, None], other=0.0).to(tl.float32)'''
assert old in part
part=part.replace(old,'''        vraw = _partial_load(qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride + (HQ + HKV + hblk * HPB) * D + dcol[None, :],jm[:,None],PS,PSS).to(tl.float32)''')
a0=part.index('        gate_pre = tl.load(');z0=part.index('\n\n',a0)
part=part[:a0]+'''        if PWIDTH == (HQ+2*HKV)*D:
            gate_pre=tl.full((ROWS,D),float('inf'),tl.float32)
        else:
            gate_pre=_partial_load(qkv_ptr+tok.to(tl.int64)[:,None]*qkv_stride+(HQ+2*HKV)*D+qh[:,None]*D+dgt[None,:],rvalid[:,None],PS,PSS)
'''+part[z0:]
helper='''@triton.jit
def _partial_load(ptr,mask,S:tl.constexpr,SS:tl.constexpr):
 v=tl.load(ptr,mask,other=0).to(tl.float32)
 if S>0:
  for i in tl.static_range(1,S):v+=tl.load(ptr+i*SS,mask,other=0).to(tl.float32)
 return v.to(tl.bfloat16).to(tl.float32)

'''
op=op[:a]+helper+part+op[z:]
op=op.replace('GPRE=fc.get("gate_pre", True),','GPRE=True,PS=_DIRECT_PS,PSS=_DIRECT_SS,PWIDTH=_DIRECT_WIDTH,')
op+='''
_DIRECT_PS=0;_DIRECT_SS=0;_DIRECT_WIDTH=2304
def attention_pair(pair,data):
 global _DIRECT_PS,_DIRECT_SS,_DIRECT_WIDTH
 p,s=pair;_DIRECT_PS=s if p.dtype==torch.float32 else 0;_DIRECT_SS=p.stride(0);_DIRECT_WIDTH=p.shape[-1]
 c=data['config'];assert c['window'] and c['num_q_heads']==16 and c['page_size']==1
 cs=_cos_sin(c['rope_theta'],64,p.device)
 return _fused(data,c,p[0],c['batch']*c['q_len'],cs)
'''
(o/'operator_direct.py').write_text(op)
# Owned virtual packed-QKV loads: sum FP32 planes in input order, round BF16,
# then execute the donor's unchanged normalization/rope/attention/merge math.
cu=(P/'lee_kernels.cu.inc').read_text()
helper=r'''
struct QView {const void* data;int S,W,RS;int64_t SS;};
static QView view(torch::Tensor p){return {p.data_ptr(),p.scalar_type()==torch::kFloat32?(int)p.size(0):0,(int)p.size(2),(int)p.stride(1),p.stride(0)};}
template<typename V,typename Params>
__device__ __forceinline__ V qread(const Params& p,const __nv_bfloat16* src){
 const int64_t off=src-p.qkv;const int tok=off/2304,c=off%2304;
 if(p.qv.S==0 && p.qv.W==2304)return *(const V*)src;
 V out;__nv_bfloat16* dst=(__nv_bfloat16*)&out;
 #pragma unroll
 for(int i=0;i<(int)(sizeof(V)/2);i++){
  float x=INFINITY;
  if(c+i<p.qv.W){
   const int64_t at=(int64_t)tok*p.qv.RS+c+i;
   if(p.qv.S){const float* fp=(const float*)p.qv.data; x=fp[at];
    #pragma unroll 1
    for(int s=1;s<p.qv.S;s++)x+=fp[s*p.qv.SS+at];
   }else x=__bfloat162float(((const __nv_bfloat16*)p.qv.data)[at]);
  }
  dst[i]=__float2bfloat16_rn(x);
 }
 return out;
}
'''
cu=cu.replace('struct WF {',helper+'\nstruct WF {',1)
at=cu.index('struct WF {');z=cu.index('\n};',at);cu=cu[:z]+'\n  QView qv;'+cu[z:]
at=cu.index('struct F {');z=cu.index('\n};',at);cu=cu[:z]+'\n  QView qv;'+cu[z:]
# Replace only reviewed QKV global loads, leaving FP8 history/SMEM loads alone.
changes=[]
# Work backwards so source boundaries remain stable under replacements.
bound=cu.index('struct F {')
def edits(s,param):
 pattern=re.compile(r'\*\(const (uint[24])\*\)\(');out=[];end=0
 for m in pattern.finditer(s):
  j=m.end();depth=1
  while depth:
   if s[j]=='(':depth+=1
   elif s[j]==')':depth-=1
   j+=1
  expr=s[m.end():j-1]
  if re.match(r'(src|krow|row0|row)\b',expr):
   out.append(s[end:m.start()]+f'qread<{m.group(1)}>({param},{expr})');end=j;changes.append([param,m.group(1),expr])
 s=''.join(out)+s[end:]
 pattern=re.compile(r'__ldg\(\(const (uint[24])\*\)\(');out=[];end=0
 for m in pattern.finditer(s):
  j=m.end();depth=1
  while depth:
   if s[j]=='(':depth+=1
   elif s[j]==')':depth-=1
   j+=1
  expr=s[m.end():j-1]
  if re.match(r'(f\.qkv|p\.qkv|row0)\b',expr):
   assert s[j]==')';out.append(s[end:m.start()]+f'qread<{m.group(1)}>({param},{expr})');end=j+1;changes.append([param,m.group(1),expr])
 return ''.join(out)+s[end:]
cu=edits(cu[:bound],'f')+edits(cu[bound:],'p')
cu=cu.replace('__bfloat162float(krow[(16 + kh) * D + d0 + i])','__bfloat162float(qread<__nv_bfloat16>(f,krow+(16+kh)*D+d0+i))').replace('__bfloat162float(krow[(18 + kh) * D + d0 + i])','__bfloat162float(qread<__nv_bfloat16>(f,krow+(18+kh)*D+d0+i))')
cu=cu.replace('qraw[h][0] = qp[0]; qraw[h][1] = qp[1];','qraw[h][0] = qread<uint4>(p,(const __nv_bfloat16*)qp); qraw[h][1] = qread<uint4>(p,(const __nv_bfloat16*)(qp+1));')
# gate4's local bb arrays stay ordinary register reads; only its global else.
cu=cu.replace('gv[i] = __bfloat162float(gt[i]);','gv[i] = __bfloat162float(qread<__nv_bfloat16>(p,gt+i));')
cu=cu.replace('  p.qkv = (const __nv_bfloat16*)qkv.data_ptr();','  p.qv=view(qkv);p.qkv = (const __nv_bfloat16*)qkv.data_ptr();',1)
host=(P/'lee_host.cu.inc').read_text().replace('L=qkv.size(1)','L=qkv.size(1)/B').replace('  int dev,sms;','  f.qv=view(qkv);\n  int dev,sms;')
# Native one-lane/2CTA experiment remains an independent launcher option later.
a0=host.index('   if(x==68)');z0=host.index('\n  }\n }',a0)
flags=sorted({x|bit for x in (68,324,32836) for bit in (0,128,32768,2048,128|32768|2048)})
host=host[:a0]+"   switch(x){\n"+''.join(f'    case {x}:apf<nl,{x}>(grid,smp+{(1 if x&128 else 0)}*nl*2048,st,a,f);break;\n' for x in flags)+'    default:TORCH_CHECK(false,"unknown hybrid flag");\n   }'+host[z0:]
# A negative launcher kind means force K-major attnd for full decode.
host=host.replace('if(kind==1){','if(kind==1 || kind==-1){')
cu+='\n'+host;(o/'lee_hybrid.cu').write_text(cu)
run=(P/'lee_runtime.py.inc').read_text().replace('def attention(qkv,data):','def attention_pair(pair,data,arm="F"):\n p,sp=pair;qkv=p')
run=run.replace('if L==1 and sl and B in (16,32):','if L==1 and (sl or arm=="K") and B in (16,32):').replace('kind=1;nl=','kind=(-1 if arm=="K" else 1);nl=')
run=run.replace('key=(kind,B,L,ns)','''if L==8 and kind==3:
  if arm=="P":x|=128
  if arm=="Q":x|=32768
  if arm=="E":x|=2048
  if arm=="H":x|=128|32768|2048
 key=(kind,B,L,ns)''').replace('qkv.reshape(B,L,2304)','qkv')
run=run.replace("name='p5b_lee_'","name='p5b_hybrid_'")
(o/'lee_hybrid.py').write_text('CUDA='+repr(cu)+'\n'+run)
meta=dict(operator=hashlib.sha256(op.encode()).hexdigest(),lee=hashlib.sha256(cu.encode()).hexdigest(),lee_runtime=hashlib.sha256(run.encode()).hexdigest(),qkv_load_replacements=changes,notices_sha256=hashlib.sha256((P/'NOTICES.txt').read_bytes()).hexdigest())
(o/'kernel_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:v for k,v in meta.items() if k!='qkv_load_replacements'})
