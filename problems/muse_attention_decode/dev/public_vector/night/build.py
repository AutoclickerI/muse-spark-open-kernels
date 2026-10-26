"""Outcome-specific production candidates on frozen S22; reviewed licensed sources."""
import ast,base64,difflib,hashlib,importlib.util,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;V=H.parent;R=H.parents[4];O=H/'out';O.mkdir(exist_ok=True)
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def load(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
pv=load(V/'build.py','night_vector_build')
PORTS={'d32S':(32,1,True),'v1S':(1,8,True),'v1F':(1,8,False),'v16S':(16,8,True),'v16F':(16,8,False),'v32S':(32,8,True)}
VERIFY={k:v for k,v in PORTS.items() if k not in ('d32S','v1S')}
ALL={**PORTS,'d1S':(1,1,True),'d1F':(1,1,False)}

def stage_source(source):
 cu=pv.assign(source,'CUDA')
 # Only verify families are used in this extension.
 a=cu.index('__device__ __forceinline__ void wg_n16_qk(');z=cu.index('\nvoid lee_attention(',a)
 cu=cu[:a]+cu[z:]
 cu=cu.replace('int S,W,RS;int64_t SS;','int S,W,RS;int64_t SS;uint32_t stage;int first;')
 needle=' const int64_t off=src-p.qkv;const int tok=off/2304,c=off%2304;'
 assert needle in cu
 cu=cu.replace(needle,needle+'''
 if(p.qv.stage && c+(int)sizeof(V)/2<=1280){
  const uint32_t at=p.qv.stage+((tok-p.qv.first)*1280+c)*2;
  V v;uint32_t* q=(uint32_t*)&v;
  if constexpr(sizeof(V)==16)asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(q[0]),"=r"(q[1]),"=r"(q[2]),"=r"(q[3]):"r"(at):"memory");
  else asm volatile("ld.shared.v2.u32 {%0,%1},[%2];":"=r"(q[0]),"=r"(q[1]):"r"(at):"memory");
  return v;
 }
''')
 # Setmaxnreg transfers permit two 384-thread NL1 CTAs: initial <=80 regs,
 # producer32, consumers112 -> 32,768 registers/CTA. Native occupancy is read.
 cu=cu.replace('__launch_bounds__(128 * (2 * NL + 1), 1)', '__launch_bounds__(128 * (2 * NL + 1), (NL==1?2:1))')
 needle='  pv_wait();\n  if (EQ && (!shareQ || LN==0))'
 assert needle in cu
 stage='''  pv_wait();
  const uint32_t rawq=shareQ?sb+RS*16384+8192:vf0;
  // V-widening storage is dead until the first mainloop iteration. Keep the
  // query tile, history ring and producer page-table queue disjoint.
  if(!shareQ || LN==0){
    for(int i=tid;i<8*1280/4;i+=256){
      const int tok=i/(1280/4),c=(i%(1280/4))*4;
      float4 v=make_float4(0,0,0,0);
      if(f.qv.S){
        const float* p=(const float*)f.qv.data+(int64_t)(b*8+tok)*f.qv.RS+c;
        #pragma unroll 1
        for(int s=0;s<f.qv.S;s++){
          const float4 x=*(const float4*)(p+(int64_t)s*f.qv.SS);
          v.x+=x.x;v.y+=x.y;v.z+=x.z;v.w+=x.w;
        }
      }else{
        const __nv_bfloat162* p=(const __nv_bfloat162*)((const __nv_bfloat16*)f.qv.data+(int64_t)(b*8+tok)*f.qv.RS+c);
        const float2 x=__bfloat1622float2(p[0]),y=__bfloat1622float2(p[1]);v=make_float4(x.x,x.y,y.x,y.y);
      }
      const __nv_bfloat162 x=__floats2bfloat162_rn(v.x,v.y),y=__floats2bfloat162_rn(v.z,v.w);
      asm volatile("st.shared.v2.u32 [%0],{%1,%2};"::"r"(rawq+i*8),"r"(*(const uint32_t*)&x),"r"(*(const uint32_t*)&y):"memory");
    }
  }
  if(shareQ)asm volatile("bar.sync 6, %0;"::"n"(256*NL):"memory");
  else asm volatile("bar.sync %0,256;"::"r"(6+LN):"memory");
  f.qv.stage=rawq;f.qv.first=b*8;
  if (EQ && (!shareQ || LN==0))'''
 cu=cu.replace(needle,stage)
 # Release this temporary storage only when every reader of that request is
 # finished. Idle balanced lanes never join a live lane's barrier.
 needle='  fence_async();\n  asm volatile("bar.sync %0, 128;\\n" :: "r"(wbar) : "memory");\n  PS(2);'
 assert needle in cu
 cu=cu.replace(needle,'''  f.qv.stage=0;
  if(shareQ)asm volatile("bar.sync 6, %0;"::"n"(256*NL):"memory");
  else asm volatile("bar.sync %0,256;"::"r"(6+LN):"memory");
'''+needle,1)
 # The staged query no longer needs the EQ live preload. Fifty percent of the
 # fast-path exps use the existing degree3 FP32 approximation in the issue arm.
 cu=cu.replace('(PEX && (j & 3) == 0)','(PEX && (j & 1) == 0)')
 a=cu.index('\nvoid lee_attention(')
 cu=cu[:a]+(H/'stage_host.cu.inc').read_text()
 run=(V.parent/'public_ports/lee_runtime.py.inc').read_text();run=run[:run.index('def attention(')]
 run=run.replace("name='p5b_lee_'","name='p5b_night_stage_'").replace("functions=['lee_attention']","functions=['lee_attention','stage_resources']")
 run=run.replace("_LOCAL=False","_LOCAL=False\n_STAGE_MODE=3")
 run=run.replace("DECL='void lee_attention", "DECL='std::vector<int64_t> stage_resources(int64_t,int64_t);\\nvoid lee_attention")
 run+=(H/'stage_runtime.py.inc').read_text()
 return 'CUDA='+repr(cu)+'\n'+run,cu

def main():
 # Snapshot the newest P4 source, with a strict identity before construction.
 path=R/'problems/muse_decode_layer/cand_r3_best7_s22_final.py';packed=path.read_text();base=pv.unpack(packed)
 assert sha(base)=='4b8c803db080f4e0f2a69e4c666ee91573fc6ffde238d2e63db4223f83f84d27'
 assert sha(packed)=='d1ccdc358ef5fb0f038a1d0e741465f512f498e90d39a16e56f3f0b0e47615f3'
 b=json.loads((V/'out/bundle.json').read_text());stage,cu=stage_source(b['modules']['lee_F'])
 modules={'lee_D':b['modules']['lee_D'],'op_D':b['modules']['op_D'],'lee_F':b['modules']['lee_F'],'op_F':b['modules']['op_F'],'lee_stage':stage,'pack':b['modules']['pack']}
 policy={'INC':(ALL,'F'),'CHAIN':(PORTS,'D'),'STAGE':(VERIFY,'STAGE'),'LANE1':(VERIFY,'LANE1'),'SOFTMAX':(VERIFY,'SOFTMAX')}
 arms={};tails={}
 for name,(cells,mode) in policy.items():
  selection={v:mode for v in cells.values()};need=set()
  for key in selection:
   if key in ((1,1,True),(1,1,False)):continue
   if mode in ('STAGE','LANE1','SOFTMAX'):need.add('lee_stage')
   else:need.add(('op_' if key==(1,8,True) else 'lee_')+mode)
  if mode=='D':need.add('pack')
  tail='\n_NIGHT_SELECTION='+repr(selection)+'\n_NIGHT_SOURCES='+repr({n:modules[n] for n in sorted(need)})+'\n'+(H/'runtime.py.inc').read_text()
  if name=='INC':tail+='\n_s16_global_C128_cs_CUDA='+repr(b['native_cs'])+'\n_s16_global_C128_cs__MOD={}\nPARTIAL_CUDA='+repr(b['native_partial'])+'\n_PP_MOD={}\n'
  source=base+tail;compile(source,'night_'+name,'exec');tails[name]=tail
  arms[name]=dict(sha256=sha(source),rows=list(cells),mode=mode)
  # Packed exact candidate; do not retain five full copies of the base in git.
  header=packed[:packed.index('import torch,base64')]
  enc=base64.b85encode(lzma.compress(source.encode(),preset=9)).decode()
  shipment=header+'import torch,base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(enc)+')).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=='+repr(sha(source))+'\n_file="<p5b-night-'+sha(source)[:12]+'>"\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,"exec"),globals())\n'
  (O/(name+'.py')).write_text(shipment);arms[name]['shipped_sha256']=sha(shipment)
  patch=''.join(difflib.unified_diff(base.splitlines(True),source.splitlines(True),fromfile='a/'+str(path.relative_to(R)),tofile='b/'+str(path.relative_to(R))))
  (H/(name+'.patch')).write_text(patch);arms[name]['patch_sha256']=sha(patch)
 # A library successor can defeat attention's entry release even after the
 # attention producer itself joins PDL. Reuse the existing notice-preserved NF
 # raw-BF16 O128/O256 kernels, without changing attention or their math.
 tail='''
NF_PORT_SELECTED |= {(128,'o'),(256,'o')}
_night_o_delta=_s16_prepare_delta
def _s16_prepare_delta(weights,state):
    state=_night_o_delta(weights,state)
    if not {(128,'o'),(256,'o')} <= state['nf_port']['plans'].keys():
        state['nf_port']=nf_prepare(weights,NF_PORT_SELECTED)
    return state
'''
 source=base+tail;compile(source,'night_OCHAIN','exec');tails['OCHAIN']=tail
 arms['OCHAIN']=dict(sha256=sha(source),rows=['v16S','v16F','v32S','v32F'],mode='OCHAIN')
 header=packed[:packed.index('import torch,base64')];enc=base64.b85encode(lzma.compress(source.encode(),preset=9)).decode()
 shipment=header+'import torch,base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(enc)+')).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=='+repr(sha(source))+'\n_file="<p5b-night-'+sha(source)[:12]+'>"\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,"exec"),globals())\n'
 (O/'OCHAIN.py').write_text(shipment);arms['OCHAIN']['shipped_sha256']=sha(shipment)
 patch=''.join(difflib.unified_diff(base.splitlines(True),source.splitlines(True),fromfile='a/'+str(path.relative_to(R)),tofile='b/'+str(path.relative_to(R))))
 (H/'OCHAIN.patch').write_text(patch);arms['OCHAIN']['patch_sha256']=sha(patch)
 # P4's primary trace protocol includes INC at every shape, including cells
 # where it is an exact dispatch twin of S22.
 arms['INC']['rows']=[]
 (O/'stage.cu').write_text(cu);(O/'stage.py').write_text(stage)
 (O/'bundle.json').write_text(json.dumps(dict(base=base,base_packed=packed,modules=modules,tails=tails)))
 (H/'manifest.json').write_text(json.dumps(dict(base_sha256=sha(base),base_shipped_sha256=sha(packed),arms=arms,modules={k:sha(v) for k,v in modules.items()},stage_cuda_sha256=sha(cu)),indent=2)+'\n');print({n:a['sha256'][:8] for n,a in arms.items()})
if __name__=='__main__':main()
