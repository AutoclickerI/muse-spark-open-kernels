"""Causal activation-transport variants against S13 and the already-measured BUNDLE arm."""
from pathlib import Path
import ast,hashlib
from build_reuse_cache import source
def act_source():return source(False)
def decoded_source():return source(True)
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def build():
 src=(ROOT/'prof/lab/moe_nt1_reuse_phase.py').read_text()
 driver=(DEV/'reuse_phase_lab.py.inc').read_text()
 a=driver.index('PHASE_ARMS=');b=driver.index('\ndef phase_summary(',a)
 driver=driver[:a]+'''PHASE_ARMS={'BUNDLE':2,'COMPACT':2,'PREFETCH':2}
def active_module(arm):return {'BUNDLE':reusep_module,'COMPACT':reusec_module,'PREFETCH':reusef_module}[arm]()
def active_geometry(state,m,arm,trace=False):return {'BUNDLE':reusep_geometry,'COMPACT':reusec_geometry,'PREFETCH':reusef_geometry}[arm](state,m,2,trace)
def phase_routed(pr,rs,state,arm,trace=False,stamps=None):
 x,ids,tw,lids=route_call(pr,rs);d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 if arm=='S13':out=smallm_experts(d,state,69)
 else:out={'BUNDLE':reusep_experts,'COMPACT':reusec_experts,'PREFETCH':reusef_experts}[arm](d,state,2,trace,stamps)
 return out,x,ids,tw,lids
'''+driver[b:]
 driver=driver.replace('(smallm_module,reuse_module,reusep_module)', '(smallm_module,reusep_module,reusec_module,reusef_module)').replace('max_workers=3','max_workers=4')
 driver=driver.replace('reuse_bind(state);reusep_bind(state)', 'reusep_bind(state);reusec_bind(state);reusef_bind(state)')
 a=driver.index('    for mode in (0,2):');b=driver.index('   for m in (8,16,32):',a)
 driver=driver[:a]+'''    for arm in PHASE_ARMS:
     for trace in (0,1):lines.append(f'GEOM {arm} trace={trace} grid,regs,localB,smemB,resident='+str(active_geometry(state,32,arm,trace)))
'''+driver[b:]
 driver=driver.replace("if all(a in graphs for a in ('BASE','BUNDLE')):", "if all(a in graphs for a in ('BUNDLE','PREFETCH')):")
 driver=driver.replace("timeline_timing(graphs['BASE'],graphs['BUNDLE'])", "timeline_timing(graphs['BUNDLE'],graphs['PREFETCH'])").replace("'BUNDLE/BASE'", "'PREFETCH/BUNDLE'")
 driver=driver.replace("if m==32:", "if m==32 and seed<8:").replace("for arm,mode in (('BASE',0),('BUNDLE',2)):", "for arm,mode in (('BUNDLE',2),('COMPACT',2),('PREFETCH',2)):")
 driver=driver.replace('grid=reusep_geometry(state,m,mode,True)[0]', 'grid=active_geometry(state,m,arm,True)[0]')
 driver=driver.replace('BASE reproduces ONCE math, SPEC block-count specialization + BUNDLE parallel multi-block reduction/epilogues; exact original ONCE retained as OLD control.', 'BUNDLE prior arm vs ACT extra-token activation copies issued before dequant,2 per-thread shared buffers; DECODED also stages decoded A transiently in the reduction workspace,fully register-caches B0,and bundles epilogues. Dequant exactly once per tile/g in all arms. This keeps8-way Ksplit and2 H200 CTAs/SM.')
 driver=driver.replace('TRACE separately compiled: globaltimer only at phase/unit boundaries,no per-g clocks.8B trace spills vs176B old; report observer/off ratios. Trace caches4 extra operand words in shared; same2 native H200 CTAs/SM. Only local BUNDLE TRACE uses1 stage to fit sm120.', 'TRACE separately compiled: phase/unit boundary globaltimer; some native builds spill (resource table); report observer/off ratios. Native shared/regs/residency are queried. Optimized ACT/DECODED local stand-ins use1 raw stage to fit sm120; native H200 uses2. BUNDLE local TRACE alone also uses1.')
 driver=driver.replace('BUNDLE prior arm vs ACT extra-token activation copies issued before dequant,2 per-thread shared buffers; DECODED also stages decoded A transiently in the reduction workspace,fully register-caches B0,and bundles epilogues. Dequant exactly once per tile/g in all arms. This keeps8-way Ksplit and2 H200 CTAs/SM.', 'BUNDLE prior arm vs COMPACT: route prefixes reuse reduction scratch and one B0 word moves to a register. M32 dynamic99328B;2 CTAs +1KiB reserved each exactly196KiB. Hopper preferred carveout85pct. PREFETCH additionally issues extra-block L1 prefetches before dequant and uses ordinary cached activation loads; acquisition precedes W2 reads. Same arithmetic/8-way Ksplit/2 raw stages; prepare unchanged.')
 driver=driver.replace('Optimized ACT/DECODED local stand-ins use1 raw stage to fit sm120; native H200 uses2. BUNDLE local TRACE alone also uses1.', 'All untraced arms execute native2 stages locally. Only traced local arms use1 stage to fit lower shared capacity; H200 uses2. Registers/spills queried, trace overhead measured.')
 src+='\n'+act_source()[0]+'\n'+decoded_source()[0]+'\n'+driver
 p=ROOT/'prof/lab/moe_nt1_cache.py';p.write_text(src);ast.parse(src);print(p,hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
