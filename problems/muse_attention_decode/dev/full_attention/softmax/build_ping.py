"""Alternate two consumer WGs: QK/PV for one, softmax for the other; producer unchanged."""
import ast
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent
issue=r'''
template<int ST> DI void ping_issue(const VP&p,uint8_t*Q,uint8_t*Ks,uint8_t*VT,uint64_t*full,int hh,int rr,
 float (&O)[8][4],float (&sx)[8][4],uint32_t (&pc)[8]){
 mb_wait(&full[(rr+1)%ST],((rr+1)/ST)&1);
 protect(O);protect(sx);protect(pc);CLK_MARK(ti);wg_fence();
 qk(sx,Q,Ks+((rr+1)%ST)*8192+hh*4096);wg_commit();
 wg_fence();pv(O,pc,VT+(rr%ST)*8192+hh*4096);wg_commit();
 CLK_ADD(0,51,ti);CLK_MARK(tw);
 // Each MMA phase completes before its cross-WG boundary. Keeping O pending
 // across a reconvergent C++ branch causes C7515 accumulator lifetime hazards.
 wg_wait<0>();protect(sx);protect(O);protect(pc);CLK_ADD(0,48,tw);
}
template<int ST> DI void ping_drain(uint64_t*empty,int rr,float (&O)[8][4],uint32_t (&pc)[8]){
 protect(O);protect(pc);__syncwarp();
 if((threadIdx.x&31)==0)mb_arrive(&empty[rr%ST]);
}
template<int ST> DI void ping_last(uint8_t*VT,uint64_t*empty,int hh,int rr,float (&O)[8][4],uint32_t (&pc)[8]){
 protect(O);protect(pc);wg_fence();pv(O,pc,VT+(rr%ST)*8192+hh*4096);wg_commit();wg_wait<0>();ping_drain<ST>(empty,rr,O,pc);
}
template<int ST,bool MASK> DI void ping_step(const VP&p,uint8_t*Q,uint8_t*Ks,uint8_t*VT,uint64_t*full,uint64_t*empty,
 int hh,int rr,bool more,float (&O)[8][4],float (&sx)[8][4],uint32_t (&pc)[8],uint32_t (&pn)[8],float (&m)[2],float (&l)[2],
 float sc,int kbase,const int(&lo)[2],const int(&hi)[2]){
 float alpha[2];
 if(hh==0){
  ping_issue<ST>(p,Q,Ks,VT,full,hh,rr,O,sx,pc);
  // Release the previous V slot before the cross-WG phase barrier. Otherwise
  // ST2 deadlocks when WG1 waits for tile rr+2 while the producer needs slot rr.
  ping_drain<ST>(empty,rr,O,pc);
 }else{
  // WG1 issued QK(rr+1)/PV(rr) in the preceding half-phase.
  CLK_MARK(ts);softmax<MASK>(sx,m,l,alpha,pn,sc,kbase,lo,hi,true);CLK_ADD(0,49,ts);
  ping_drain<ST>(empty,rr,O,pc);rescale(O,alpha);
 }
 asm volatile("bar.sync 3,256;":::"memory");
 if(hh==0){
  CLK_MARK(ts);softmax<MASK>(sx,m,l,alpha,pn,sc,kbase,lo,hi,true);CLK_ADD(0,49,ts);rescale(O,alpha);
 }else{
  if(more)ping_issue<ST>(p,Q,Ks,VT,full,hh,rr+1,O,sx,pn);
  else ping_last<ST>(VT,empty,hh,rr+1,O,pn);
 }
 asm volatile("bar.sync 3,256;":::"memory");
}
'''
for parent,name in [('ref','ping'),('mix25','ping25')]:
 s=(H/(parent+'.cu')).read_text();a=s.index('template <int ST, bool MASK>\nDI void step');s=s[:a]+issue+'\n'+s[a:]
 a=s.index('      // Steps j:');b=s.index('    r += nr;',a)
 loop=r'''
      if(nr>1 && hh==1)ping_issue<ST>(p,Q,Ks,VT,full,hh,r,O,sx,pa);
      const bool lastmask=nr>=2 && ka+nr==nc;
      const int nplain=nr-1-(lastmask?1:0);int j=0;
      for(;j+2<=nplain;j+=2){
       ping_step<ST,false>(p,Q,Ks,VT,full,empty,hh,r+j,j+2<nr,O,sx,pa,pb,m,l,sc,lo+64*(ka+j+1),lo_r,hi_r);
       ping_step<ST,false>(p,Q,Ks,VT,full,empty,hh,r+j+1,j+3<nr,O,sx,pb,pa,m,l,sc,lo+64*(ka+j+2),lo_r,hi_r);
      }
      if(nplain&1){
       ping_step<ST,false>(p,Q,Ks,VT,full,empty,hh,r+j,j+2<nr,O,sx,pa,pb,m,l,sc,lo+64*(ka+j+1),lo_r,hi_r);
       if(lastmask){
        ping_step<ST,true>(p,Q,Ks,VT,full,empty,hh,r+nr-2,false,O,sx,pb,pa,m,l,sc,lo+64*(nc-1),lo_r,hi_r);
        if(hh==0)ping_last<ST>(VT,empty,hh,r+nr-1,O,pa);
       }else if(hh==0 || nr==1)ping_last<ST>(VT,empty,hh,r+nr-1,O,pb);
      }else{
       if(lastmask){
        ping_step<ST,true>(p,Q,Ks,VT,full,empty,hh,r+nr-2,false,O,sx,pa,pb,m,l,sc,lo+64*(nc-1),lo_r,hi_r);
        if(hh==0)ping_last<ST>(VT,empty,hh,r+nr-1,O,pb);
       }else if(hh==0 || nr==1)ping_last<ST>(VT,empty,hh,r+nr-1,O,pa);
      }
    }
'''
 s=s[:a]+loop+s[b:]
 (H/(name+'.cu')).write_text(s)
 tree=ast.parse((H/(parent+'.py')).read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
 drv=ast.unparse(tree)+'\n';(H/(name+'.py')).write_text(drv)
 print(name,hashlib.sha256(s.encode()).hexdigest())
