"""Build a reviewable experimental patch onto frozen S13 final958f1258; no promotion is implied."""
import argparse,ast,difflib,hashlib,json
from pathlib import Path
from build_reuse_phase import source as bundle_source
from build_reuse_act import source as act_source
from build_reuse_pipe import source as decoded_source
from build_reuse_cache import source as cache_source
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best7_s13_final.py'
BUILDERS={'bundle':(bundle_source,'reusep'),'act':(act_source,'reusea'),'decoded':(decoded_source,'reused'),
 'compact':(lambda:cache_source(False),'reusec'),'prefetch':(lambda:cache_source(True),'reusef')}
def patch(base,kind,modes):
 assert hashlib.sha256(base.encode()).hexdigest()=='958f1258908f44834ec6cf0c33af00ff6c08ea62b2f5cc483a36f0a91f39d0b6'
 assert modes and set(modes)<={1,8,16,32} and all(x==2 for x in modes.values())
 build,prefix=BUILDERS[kind];cell=build()[0]
 helper='''
# P3a experimental Hopper dispatch; local forced screens set this before prepare().
_P3A_HOPPER = torch.cuda.get_device_capability() == (9, 0)
_P3A_MODES = '''+repr(modes)+'''
def _p3a_call(data,state):
    m=data['hidden_states'].shape[0];bank=state['s4_nt1_bank']
    if _P3A_HOPPER and m in _P3A_MODES:return '''+prefix+'''_experts(data,bank,_P3A_MODES[m])
    return smallm_experts(data,bank,SMALLM_FLAGS[m])

'''
 point=base.index('def _s4_prepare_banks(')
 out=base[:point]+cell+'\n'+helper+base[point:]
 old="if name=='nt1_lop3':return smallm_experts(data,state['s4_nt1_bank'],SMALLM_FLAGS[data['hidden_states'].shape[0]])"
 assert out.count(old)==1;out=out.replace(old,"if name=='nt1_lop3':return _p3a_call(data,state)",1)
 old="    smallm_bind(state['s4_nt1_bank'],SMALLM_FLAGS)"
 assert out.count(old)==1;out=out.replace(old,old+"\n    if _P3A_HOPPER:"+prefix+"_bind(state['s4_nt1_bank'])",1)
 ast.parse(out)
 return out,cell

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kind',choices=BUILDERS,required=True)
 ap.add_argument('--map',default='32:2');ap.add_argument('-o',type=Path,required=True);args=ap.parse_args()
 modes=dict(tuple(map(int,r.split(':'))) for r in args.map.split(','));base=BASE.read_text();cand,cell=patch(base,args.kind,modes)
 args.o.parent.mkdir(parents=True,exist_ok=True);rel=str(BASE.relative_to(ROOT))
 complete={m:(args.kind+':'+str(modes[m]) if m in modes else f'S13:{95 if m==1 else 69}') for m in (1,8,16,32)}
 files={args.o:cand,args.o.with_suffix('.cell.py'):cell,args.o.with_suffix('.patch'):''.join(difflib.unified_diff(base.splitlines(True),cand.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))}
 for p,s in files.items():p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest()[:8])
 meta={'base_sha256':hashlib.sha256(base.encode()).hexdigest(),'cell_kind':args.kind,'complete_map':complete,'candidate_sha256':hashlib.sha256(cand.encode()).hexdigest(),'promoted':False}
 args.o.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print('complete map',complete)
if __name__=='__main__':main()
