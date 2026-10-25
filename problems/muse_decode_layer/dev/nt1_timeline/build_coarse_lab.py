"""Pair coarse WGMMA pipelines and producer-fused activation permutation against exact S11."""
from pathlib import Path
import ast,hashlib
from build_wg_coarse import source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]

def build():
 src=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text()
 original=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text()
 a=original.index('@triton.jit\ndef timeline_route(');b=original.index('\ndef route_call(',a)
 router=original[a:b].replace('timeline_route(', 'timeline_route_perm(')
 old='    tl.store(X+t*4096+d,norm.to(tl.bfloat16),d//(4096//REPLICAS)==part)'
 new='''    physical=(d&(~63))|(((d>>2)&3)*16+((d>>4)&3)*2+((d>>1)&1)*8+(d&1))
    tl.store(X+t*4096+physical,norm.to(tl.bfloat16),d//(4096//REPLICAS)==part)'''
 assert router.count(old)==1;router=router.replace(old,new)
 src+='\n'+source(2)[0]+'\n'+source(4)[0]+'\n'+router
 src+='''
def coarse_route(pr,rs):
 p,s=pr;m=p.shape[1];dev=p.device;x=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
 ids=torch.empty((m,8),device=dev,dtype=torch.int32);lids=torch.empty_like(ids);tw=torch.empty((m,8),device=dev)
 timeline_route_perm[(m,1)](p,s,p.stride(0),p.stride(1),rs['bias'],rs['pre'],rs['start'],x,ids,tw,lids,REPLICAS=1,USE_PDL=True,launch_pdl=True,num_warps=4 if m==8 else 8)
 return x,ids,tw,lids

def coarse_gate(output,w,truths,pr,rs,arm):
 if arm!='S11':
  out,x,ids,tw,lids=output;d=torch.arange(4096,device=x.device)
  q=(d&(~63))|(((d>>2)&3)*16+((d>>4)&3)*2+((d>>1)&1)*8+(d&1))
  output=(out,x[:,q].contiguous(),ids,tw,lids)
 return reuse_gate(output,w,truths,pr,rs)
'''
 driver=(DEV/'wg_lab.py.inc').read_text()
 driver=driver.replace('def wg_routed(', 'def coarse_routed(')
 driver=driver.replace(' x,ids,tw,lids=route_call(pr,rs);', " x,ids,tw,lids=route_call(pr,rs) if arm=='S11' else coarse_route(pr,rs);",1)
 driver=driver.replace('else wg_experts(d,state,*WG_ARMS[arm])', 'else (wg2_experts if WG_ARMS[arm][0]==2 else wg4_experts)(d,state,*WG_ARMS[arm][1:])')
 driver=driver.replace("WG_ARMS={'W4G2':(4,2),'W4G4':(4,4),'W8G2':(8,2),'W8G4':(8,4)}", "WG_ARMS={'K128S4':(2,4,4),'K128S8':(2,8,4),'K256S2':(4,2,4),'K256S4':(4,4,4)}")
 driver=driver.replace('wg_routed(', 'coarse_routed(')
 driver=driver.replace('(smallm_module,wg_module,activity_module)', '(smallm_module,wg2_module,wg4_module,activity_module)').replace('max_workers=3', 'max_workers=4')
 driver=driver.replace('wg_bind(state)', 'wg2_bind(state);wg4_bind(state)')
 driver=driver.replace("str(wg_geometry(state,32,*params))", "str((wg2_geometry if params[0]==2 else wg4_geometry)(state,32,*params[1:]))")
 driver=driver.replace('reuse_gate(graphs[arm].replay(),w,truths,pr,rs)', 'coarse_gate(graphs[arm].replay(),w,truths,pr,rs,arm)')
 driver=driver.replace('original S11 code bytes plus losslessly reordered raw u8 scale pairs', 'original S11 code bytes plus losslessly grouped raw u8 scales; K128/K256 stages; router and W13 stores emit the exact K-permuted BF16 activation layout, vector16B copies')
 driver=driver.replace('Stages4/8, requested grid2/4 CTAs/SM capped by real residency.', 'K128 stages4/8, K256 stages2/4; maximum3 CTAs/SM capped by shared capacity; no per-K64 scalar activation gathers.')
 driver=driver.replace('W4G4/W8G4 request4 but launch bound3 may cap them at3 resident CTAs/SM.', 'Requested grid4 is capped at the native resident count (at most3). Raw u8 scale and activation layouts have inverse checks.')
 src+='\n'+driver
 out=ROOT/'prof/lab/moe_nt1_wg_coarse.py';out.write_text(src);ast.parse(src)
 print(out.relative_to(ROOT),hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
