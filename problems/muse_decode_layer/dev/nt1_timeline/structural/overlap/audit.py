"""Require actual dynamic register instructions and role separation in the exact Hopper SASS."""
import ast,collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_overlap';cells=json.loads((D/'cells.json').read_text());counts=json.loads((D/'counts.json').read_text())['cells'];out={}
for name,spec in cells.items():
 if name not in counts or counts[name]['cuda_sha256']!=spec['cuda_sha256']:raise ValueError('final count not complete: '+name)
 log=(O/(name.lower()+'.compile.log')).read_text();assert 'C7508' not in log
 sass=subprocess.check_output(['cuobjdump','--dump-sass',str(O/(name.lower()+'.o'))],text=True);(O/(name.lower()+'.sass')).write_text(sass);rows={}
 for fn,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
  v=re.search(r'moe_varILi1ELi(69|325)E',fn)
  if not v:continue
  ops=lambda s:dict(collections.Counter(x.split('.')[0] for x in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)',s)))
  rr=re.search(r'Function properties for '+re.escape(fn)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers([^\n]*)',log);assert rr
  stack,stores,loads,regs=map(int,rr.groups()[:4]);sm=re.search(r'(\d+) bytes smem',rr[5]);static=int(sm[1]) if sm else 0
  part=dict(registers=regs,stack=stack,spill_stores=stores,spill_loads=loads,static_smem=static,ops=ops(b))
  if spec['kind'].startswith('WS'):
   assert 'USETMAXREG.DEALLOC.CTAPOOL 0x20' in b
   target='0xc0' if spec['kind']=='WS4' else '0x68';assert 'USETMAXREG.TRY_ALLOC.CTAPOOL UP0, '+target in b
   cut=b.index('USETMAXREG.DEALLOC');part['producer_tail_ops']=ops(b[cut:]);part['consumer_prefix_ops']=ops(b[:cut]);part['dynamic_regs']=dict(producer=32,consumer=192 if spec['kind']=='WS4' else 104)
   assert not part['producer_tail_ops'].get('HMMA',0) and not part['producer_tail_ops'].get('HMUL2',0)
   assert not part['consumer_prefix_ops'].get('UBLKCP',0) and part['producer_tail_ops'].get('UBLKCP',0)>=1
  if v[1]=='69':assert stores==loads==0
  rows[v[1]]=part
 out[name]=dict(cuda_sha256=spec['cuda_sha256'],cell_sha256=spec['cell_sha256'],flags=rows)
 (D/'sass_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(name,{k:(v['registers'],v['static_smem'],v['spill_stores']) for k,v in rows.items()},flush=True)
#The PIPE.md math count applies to currentBASE:hash its actual unchanged decode/MMA body.
P=D.parent.parent;base=R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py';src=next(ast.literal_eval(n.value) for n in ast.parse(base.read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SMALLM_VAR_CUDA')
def core(cu):
 a=cu.index('    const uint8_t* tile = stages + st * TILE_BYTES;');b=cu.index('    __syncwarp();',a);return cu[a:b]
current=(R/'runs/screen/p3a_structural/fused1.cu').read_text();assert core(src)==core(current)
prem=dict(receipts=['16c180784f414576','d650c04ead114119','29ad388de976432c'],current_BASE_math_byte_identical_to_measured_S14=True,math_sha256=hashlib.sha256(core(current).encode()).hexdigest(),per_math_warp_tile=dict(weights_per_lane=256,decode_scale_instructions=416,of_which_HMUL2=128,MMA=32,shared_weight_scale_loads=9,total=457),per_warp_segment_Bfill=dict(vector_global_loads=16,instruction_envelope=46),PAIR_epi_per_CTA_tile=dict(W13=148,W2=129,main_writer_warps=[0,1]),BASE_producer_warp=7,BASE_producer_after_all_consumer_partial_store_barrier=True,interpretation='Static instruction/model counts plus calibrated same-op rates;not measured issue-slot occupancy or additive wall-time attribution',primary_setmaxnreg='https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg')
(D/'premeasure.json').write_text(json.dumps(prem,indent=2)+'\n')
