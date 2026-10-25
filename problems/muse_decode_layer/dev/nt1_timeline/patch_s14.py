"""Replace S14's small-M cell with a checked experimental cell; preserve the complete95/69 map."""
import argparse,ast,difflib,hashlib,json
from pathlib import Path
from build_epilogue import source
from build_epilogue_pair import source as pair_source
from build_epi_next import source as next_source,KINDS as NEXT
from build_pipe_ready import source as pipe_source,KINDS as PIPE
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py'
KINDS={'base':0,'vec':1,'producer':2,'norm':3,'pair':4,**{k:5+i for i,k in enumerate(NEXT)},**{k:10+i for i,k in enumerate(PIPE)}}
def build(kind,out):
 base=BASE.read_text();assert hashlib.sha256(base.encode()).hexdigest()=='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
 mode=KINDS[kind]
 if kind in PIPE:prefix=PIPE[kind];cell=pipe_source(kind)[0]
 elif kind in NEXT:prefix=NEXT[kind];cell=next_source(kind)[0]
 else:prefix=('epib','epiv','epip','epin','epit')[mode];cell=(pair_source() if mode==4 else source(mode))[0]
 if kind in PIPE:
  cell_out=cell
  a=base.index('def _s4_prepare_banks(')
  wrapper=f"""
PIPE_KIND={kind!r}
_pipe_original_bind=smallm_bind
_pipe_original_experts=smallm_experts
def smallm_bind(state,flags_by_m):
    _pipe_original_bind(state,flags_by_m)
    {prefix}_bind(state,{{8:69,16:325}})
    return state
def smallm_experts(data,state,flags,stages=2,grid=264):
    if 1 < data['hidden_states'].shape[0] <= 32:return {prefix}_experts(data,state,flags,stages,grid)
    return _pipe_original_experts(data,state,flags,stages,grid)
"""
  candidate=base[:a]+cell.rstrip()+'\n'+wrapper+'\n'+base[a:]
 else:
  cell=cell.replace(prefix.upper()+'_FLAGS_BUILT','SMALLM_FLAGS_BUILT').replace(prefix.upper()+'_','SMALLM_VAR_').replace('_'+prefix.upper()+'_MODULE','_SMALLM_MODULE').replace(prefix+'_','smallm_')
  a=base.index('SMALLM_VAR_CUDA = ');b=base.index('def _s4_prepare_banks(',a)
  candidate=base[:a]+cell.rstrip()+'\n\n\n'+base[b:];ast.parse(candidate)
  cell_out=cell
 ast.parse(candidate)
 rel=str(BASE.relative_to(ROOT));patch=''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
 out.parent.mkdir(parents=True,exist_ok=True)
 files={out:candidate,out.with_suffix('.cell.py'):cell_out,out.with_suffix('.patch'):patch}
 for p,s in files.items():p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest()[:8])
 info={'kind':kind,'base_sha256':hashlib.sha256(base.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'cell_sha256':hashlib.sha256(cell.encode()).hexdigest(),'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'flag_map':{1:95,8:69,16:69,32:69},'promoted':False}
 out.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=KINDS,required=True);p.add_argument('-o',type=Path,required=True);a=p.parse_args();build(a.kind,a.o)
