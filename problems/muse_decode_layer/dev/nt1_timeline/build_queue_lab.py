"""Same producer-fused graphs; vary only the independent packed-weight queue depth."""
from pathlib import Path
import hashlib,ast
from build_wg_queue import source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def build():
 src=(ROOT/'prof/lab/moe_nt1_wg_coarse.py').read_text()
 a=src.index('"""P3a Hopper RS WGMMA cell:');b=src.index('@triton.jit\ndef timeline_route_perm(',a)
 src=src[:a]+source()[0]+'\n'+src[b:]
 src=src.replace("else (wg2_experts if WG_ARMS[arm][0]==2 else wg4_experts)(d,state,*WG_ARMS[arm][1:])", 'else wgq_experts(d,state,*WG_ARMS[arm])')
 src=src.replace("WG_ARMS={'K128S4':(2,4,4),'K128S8':(2,8,4),'K256S2':(4,2,4),'K256S4':(4,4,4)}", "WG_ARMS={'Q4':(4,4),'Q8':(8,4),'Q12':(12,4)}")
 src=src.replace('(smallm_module,wg2_module,wg4_module,activity_module)', '(smallm_module,wgq_module,activity_module)').replace('max_workers=4','max_workers=3')
 src=src.replace('wg2_bind(state);wg4_bind(state)','wgq_bind(state)')
 src=src.replace('str((wg2_geometry if params[0]==2 else wg4_geometry)(state,32,*params[1:]))', 'str(wgq_geometry(state,32,*params))')
 src=src.replace('K128/K256 stages; router and W13 stores', 'independent packed-weight TMA queue4/8/12 x4608B,3 activation stages x8192B; router and W13 stores')
 src=src.replace('K128 stages4/8, K256 stages2/4; maximum3 CTAs/SM capped by shared capacity;', 'One contiguous bulk TMA perK128, lossless exact raw code/scale regrouping. Maximum3 CTAs/SM capped by shared capacity;')
 src=src.replace('Raw u8 scale and activation layouts have inverse checks.', 'Raw code/scale and activation layouts have inverse checks. Every queue depth executes locally with native TMA and portable MMA; native H200 math is independently gated.')
 src=src.replace("device=next(v.device for v in weights.values() if isinstance(v,torch.Tensor))","device=torch.device('cuda')")
 out=ROOT/'prof/lab/moe_nt1_wg_queue.py';out.write_text(src);ast.parse(src);print(out,hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
