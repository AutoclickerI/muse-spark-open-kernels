"""Overlap the K128 page-row gather with the eight consumer warps.

A separate 128-thread producer group fills a ring using cp.async+mbarrier.
The 256 consumer threads retain the exact tested swap-AB MMA and last-CTA
merge. All GPU paths execute on sm_120 as well as Hopper.
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'cand_xqa_pipeline.py').read_text().split('\n_PIPE_PLAN = {}')[0]
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='X_CUDA' for t in n.targets))
c=ast.literal_eval(node.value)
helpers=r'''
DI void consumer_bar(){asm volatile("bar.sync 1, 256;":::"memory");}
DI void mb_init(uint64_t* b,int count){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(su(b)),"r"(count):"memory");}
DI void mb_arrive(uint64_t* b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(su(b)):"memory");}
DI void cp_arrive(uint64_t* b){asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(su(b)):"memory");}
DI void mb_wait(uint64_t* b,int phase){uint32_t done;do{asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2;selp.b32 %0,1,0,p;}":"=r"(done):"r"(su(b)),"r"(phase):"memory");}while(!done);}
'''
# All barriers in the old path are now consumer-group barriers. Initialization
# below is the sole all-thread barrier, before either group diverges.
c=c.replace('__syncthreads();','consumer_bar();')
anchor='template<int N> DI void wait(){asm volatile("cp.async.wait_group %0;"::"n"(N):"memory");}'
assert anchor in c;c=c.replace(anchor,anchor+'\n'+helpers,1)
c=c.replace('__launch_bounds__(256,2)','__launch_bounds__(384,1)')
start=c.index(' int rows[KT/64];');end=c.index(' // Every split prepares',start)
producer=r'''
 __shared__ __align__(16) uint64_t full[ST],empty[ST];
 if(tid<ST){mb_init(&full[tid],128);mb_init(&empty[tid],256);}
 __syncthreads();
 if(tid>=256){
  int ptid=tid-256;
  for(int i=0;i<ntiles;i++){
   int slot=i%ST;
   if(i>=ST)mb_wait(&empty[slot],((i/ST)-1)&1);
   #pragma unroll
   for(int rr=0;rr<KT/32;rr++){
    int row=ptid/4+rr*32,col=(ptid&3)*16,key=begin+i*KT+row;
    bool valid=key<end;
    int physical=valid?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):0;
    const uint8_t* kp=p.kc+(long long)physical*rowbytes+h*64+col;
    const uint8_t* vp=p.vc+(long long)physical*rowbytes+h*64+col;
    cp(sk+slot*KB+off(row,col),kp,valid);
    cp(sv+slot*KB+off(row,col),vp,valid);
   }
   commit();cp_arrive(&full[slot]);
   wait<ST-1>();
  }
  wait<0>();
  return;
 }
'''
c=c[:start]+producer+c[end:]
old='''  wait<ST-2>();consumer_bar();
  issue(i+ST-1,(i+ST-1)%ST);commit();
  load_rows(i+ST);
  tile<KT,P8>(st,sk+(i%ST)*KB,sv+(i%ST)*KB,begin+i*KT,end,seq,p.window,scale,vh);
  consumer_bar();'''
new='''  mb_wait(&full[i%ST],(i/ST)&1);
  consumer_bar();
  tile<KT,P8>(st,sk+(i%ST)*KB,sv+(i%ST)*KB,begin+i*KT,end,seq,p.window,scale,vh);
  consumer_bar();
  mb_arrive(&empty[i%ST]);'''
assert old in c;c=c.replace(old,new,1)
c=c.replace(' wait<0>();consumer_bar();',' consumer_bar();',1)
c=c.replace('<<<dim3(p.ns,p.hkv,p.batch),256,','<<<dim3(p.ns,p.hkv,p.batch),384,')
# Limit the instantiated ring sizes to locally executable, useful families.
start=c.index(' if(stages==7){',c.index('void xforward'))
end=c.index(' auto err=',start)
dispatch=[]
for st in (3,4):
 dispatch.append((' if' if st==3 else ' else')+(f'(stages=={st})' if st==3 else '')+'{')
 dispatch.append(f'  if(p.window)go<{st},false,true>(p,s);else go<{st},false,false>(p,s);')
 dispatch.append(' }')
c=c[:start]+'\n'.join(dispatch)+'\n'+c[end:]
lines=source.splitlines(True);lines[node.lineno-1:node.end_lineno]=['X_CUDA = r"""\n',c,'\n"""\n']
text=''.join(lines).replace('muse_xqa_pipeline128_sm','muse_xqa_warp_sm')
text=text.replace('muse_xqa_pipeline_sm','muse_xqa_warp_sm')
# Default real kernel executes locally; diagnostic will choose on Hopper.
text=text.replace('X_STAGES = 3','X_STAGES = 3')
(root/'dev/xqa_warp.cu').write_text(c)
(root/'dev/xqa_warp_candidate.py').write_text(text)
print('Built dev/xqa_warp_candidate.py')
