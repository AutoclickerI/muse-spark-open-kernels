"""Build a private cp.async.bulk row-gather variant from the accepted pipeline."""
from pathlib import Path
import ast,re
root=Path(__file__).resolve().parents[1]
source=(root/'cand_xqa_pipeline.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='X_CUDA' for t in n.targets))
c=ast.literal_eval(node.value)
# Bulk destinations are contiguous 64-byte rows; ldmatrix and V widening use
# this same row-major layout. Page-table and K/V source rows remain exact.
c=c.replace('DI int off(int row,int col){return row*64+((col>>4)^((row&3)^((row>>2)&1)))*16+(col&15);}', 'DI int off(int row,int col){return row*64+col;}',1)
helpers=r'''
DI void bar_init(uint64_t* p){asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"((uint32_t)__cvta_generic_to_shared(p)):"memory");}
DI void bar_expect(uint64_t* p,int bytes){asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(p)),"r"(bytes):"memory");}
DI bool bar_ready(uint32_t b,int parity){uint32_t done;asm volatile("{ .reg .pred %%p; mbarrier.try_wait.parity.shared.b64 %%p, [%1], %2; selp.b32 %0, 1, 0, %%p; }":"=r"(done):"r"(b),"r"(parity):"memory");return done!=0;}
DI void bar_wait(uint64_t* p,int parity){uint32_t b=(uint32_t)__cvta_generic_to_shared(p);while(!bar_ready(b,parity)){} }
DI void bulk_cp(void* dst,const void* src,int bytes,uint64_t* p){if(bytes)asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"l"(src),"r"(bytes),"r"((uint32_t)__cvta_generic_to_shared(p)):"memory");}
'''
c=c.replace('DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}', 'DI void commit(){asm volatile("cp.async.commit_group;":::"memory");}\n'+helpers,1)
c=c.replace('template<int ST,bool P8,bool ROPE>\n__global__', 'template<int ST,bool P8,bool ROPE,bool BULK>\n__global__',1)
c=c.replace('template<int ST,bool P8,bool ROPE>void go', 'template<int ST,bool P8,bool ROPE,bool BULK>void go',1)
c=c.replace('xqa<ST,P8,ROPE>', 'xqa<ST,P8,ROPE,BULK>',1)
c=c.replace('xqa<ST,P8,ROPE><<<', 'xqa<ST,P8,ROPE,BULK><<<',1)
c=c.replace('constexpr int smem=ST*2*KB+8192+(P8?0:2*KB);', 'constexpr int smem=ST*2*KB+8192+(P8?0:2*KB)+ST*8;',1)
c=c.replace('uint8_t* nk=nq+4096;uint8_t* nv=nk+2048;uint8_t* vh=nv+2048;', 'uint8_t* nk=nq+4096;uint8_t* nv=nk+2048;uint8_t* vh=nv+2048;uint64_t* bars=reinterpret_cast<uint64_t*>(vh+2*KB);',1)
c=c.replace('int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;', 'int b=blockIdx.z,h=blockIdx.y,sp=blockIdx.x,seq=p.seq[b];bool last=sp==p.ns-1;\n if(tid==0)for(int s=0;s<ST;s++)bar_init(&bars[s]);\n __syncthreads();',1)
start=c.index(' int rows[KT/64];')
end=c.index(' #pragma unroll\n for(int stage=0;stage<ST-1;stage++)',start)
bulk=r''' auto issue=[&](int tileid,int slot){
  int first=begin+tileid*KT;
  int valid=(tileid<ntiles)?min(KT,max(0,end-first)):0;
  if(tid==0)bar_expect(&bars[slot],valid*128);
  __syncthreads();
  if(tid<valid){
   int key=first+tid;
   int physical=pt[key>>p.lg]*p.ps+(key&(p.ps-1));
   uint64_t base=(uint64_t)physical*rowbytes+h*64;
   bulk_cp(sk+slot*KB+tid*64,p.kc+base,64,&bars[slot]);
   bulk_cp(sv+slot*KB+tid*64,p.vc+base,64,&bars[slot]);
  }
 };
'''
c=c[:start]+bulk+c[end:]
c=c.replace('for(int stage=0;stage<ST-1;stage++){load_rows(stage);issue(stage,stage);commit();}\n load_rows(ST-1);', 'for(int stage=0;stage<ST-1;stage++)issue(stage,stage);',1)
old='''  wait<ST-2>();__syncthreads();
  issue(i+ST-1,(i+ST-1)%ST);commit();
  load_rows(i+ST);
  tile<KT,P8>(st,sk+(i%ST)*KB,sv+(i%ST)*KB,begin+i*KT,end,seq,p.window,scale,vh);
  __syncthreads();'''
new='''  issue(i+ST-1,(i+ST-1)%ST);
  bar_wait(&bars[i%ST],(i/ST)&1); __syncthreads();
  tile<KT,P8>(st,sk+(i%ST)*KB,sv+(i%ST)*KB,begin+i*KT,end,seq,p.window,scale,vh);
  __syncthreads();'''
if old not in c: raise SystemExit('loop pattern missing')
c=c.replace(old,new,1)
c=c.replace(' wait<0>();__syncthreads();', ' if(ntiles){bar_wait(&bars[(ntiles-1)%ST],((ntiles-1)/ST)&1);__syncthreads();}',1)
# Dispatch all go calls to the bulk specialization.
c=re.sub(r'go<(\d+),(true|false),(true|false)>\(p,s\)',r'go<\1,\2,\3,true>(p,s)',c)
lines=source.splitlines(True);lines[node.lineno-1:node.end_lineno]=['X_CUDA = r"""\n',c,'\n"""\n']
out=''.join(lines).replace('muse_xqa_pipeline_sm','muse_xqa_bulk_sm')
(root/'cand_xqa_bulk.py').write_text(out);(root/'dev/xqa_bulk.cu').write_text(c);print('Built',root/'cand_xqa_bulk.py')
