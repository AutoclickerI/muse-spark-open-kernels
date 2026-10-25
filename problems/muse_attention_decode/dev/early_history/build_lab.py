"""One immutable attention candidate with one batched exact-S14 whole-layer history lab."""
import ast,base64,hashlib,lzma,re,sys
from pathlib import Path
import patch,history_build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def funcs(path,names):
    s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)

def boot(src,label):
    tag=hashlib.sha256(src.encode()).hexdigest();blob=base64.b85encode(lzma.compress(src.encode(),preset=9)).decode()
    return ('import base64,lzma,linecache,hashlib\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n'
            'assert hashlib.sha256(_BODY.encode()).hexdigest()=='+repr(tag)+'\n_FNAME='+repr('<'+label+'-'+tag[:12]+'>')+'\n'
            'linecache.cache[_FNAME]=(len(_BODY),None,_BODY.splitlines(True),_FNAME)\nexec(compile(_BODY,_FNAME,"exec"),globals())\n')

base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text();assert hashlib.sha256(base.encode()).hexdigest()==patch.BASE_SHA
synth='HIDDEN=8192;Q_SIZE=1024;QKV_SIZE=2304;D_EXPERT=4096;LOCAL_EXPERTS=32;N_EXPERTS=256\n'
synth+='ROUTER_STD=.045;ROUTER_ALIGN=.07;ROUTER_BIAS_STD=.055;STREAM_SEED=5354;STREAM_RANK=4\n'
synth+='BLOCK=16;E4M3_MAX=448.;SCALE_LOG2_MEAN=7.;SCALE_LOG2_STD=.4\n'
synth+=funcs(ROOT/'problems/muse_decode_layer/nvfp4.py',('random_weights',))+'\n'
synth+=funcs(ROOT/'problems/muse_decode_layer/reference.py',('_stream_basis','_router','generate_weights'))+'\n'
timeline=(ROOT/'prof/lab/last_sliding_timeline.py').read_text();nodes=ast.parse(timeline).body
activity=next(ast.get_source_segment(timeline,n) for n in nodes if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='TL_ACTIVITY' for t in n.targets))
helpers='\n\n'.join(ast.get_source_segment(timeline,n) for n in nodes if isinstance(n,ast.FunctionDef) and ((n.name.startswith('tl_') and n.name not in ('tl_prepare_selected','tl_host','tl_run')) or n.name in ('cs_copy','cs_delta','cs_profile')))
body='import gc,hashlib,json,linecache,math,statistics,time,types,base64,lzma\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
body+='EH_BASE='+repr(base)+'\n'+synth+'\n'+activity+'\nTL_REPS=5\n'+helpers+'\n'
# Freeze the public checker as an independent lab-only namespace. No judge
# imports of reference/checks/nvfp4; the timed kernels never call this code.
spec='import math,functools\nimport torch\n'
for file in ('nvfp4.py','checks.py','reference.py'):
    text=(ROOT/'problems/muse_decode_layer'/file).read_text()
    spec+='\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))+'\n'
body+='PUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\n'

for name,src in [('HIST_EARLY',patch.cell()),('HIST_LATE',patch.cell(False)),('HIST_CLOCK',patch.cell(clock=True))]:
    body+=name+"={'torch':torch}\nexec("+repr(src)+','+name+')\n'
old=patch.cell();cuda=history_build.build_last.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
oldnodes=ast.parse(old).body;lines=old.splitlines(True)
for n in reversed(oldnodes):
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CUDA','CPP'):
        lines[n.lineno-1:n.end_lineno]=[n.targets[0].id+'='+repr(cuda if n.targets[0].id=='CUDA' else cpp)+'\n']
old=''.join(lines)
body+="HIST_OLD={'torch':torch}\nexec("+repr(old)+',HIST_OLD)\n'
body+=(HERE/'lab_body.inc').read_text();compile(body,'early_history_lab_body','exec');(HERE/'out/lab_body.py').write_text(body)
payload=boot(body,'p5-early-history');(HERE/'early_history_lab.py').write_text(payload)
# Frozen attention incumbent: retain definitions, remove its import-time search.
host=(ROOT/'runs/muse_attention_decode/20261025T030917Z_submit_81cccbc19410433b.py').read_text();host=host[:host.rindex('if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:')]
lines=host.splitlines(True)
for n in reversed(ast.parse(host).body):
    if isinstance(n,ast.If) and '_lab' in ast.get_source_segment(host,n):lines[n.lineno-1:n.end_lineno]=[]
host=''.join(lines)
cell=patch.prefixed_cell()
host+='\n'+cell+'''
_EH_BASE_FN=kernel_fn
def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg=data['config']
    if cfg['batch']==1 and cfg['q_len']==1 and cfg['window']==2048 and cfg['head_dim']==64 and cfg['num_q_heads']==8*cfg['num_kv_heads']:
        return eh_attention(data,qt=4,nk=128,cs=16)
    return _EH_BASE_FN(data)
'''
sys.path.insert(0,str(ROOT/'tools'));import h200_lab
tag=hashlib.sha256(payload.encode()).hexdigest()[:16]
out=host+'\n'+h200_lab.WRAPPER.format(names='early_history '+tag,base='frozen attention81cc + compact early history B1; whole-layer lab exact S14f1c77ce6',payloads=[('early_history',tag,payload)],budget=550,limit=15000)+'\n'+h200_lab.IMPORT_WRAPPER.format(budget=550)
# Local screens execute changed kernels/glue. Full synthetic lab is explicitly
# invoked by check_lab.py; import-time lab runs only on actual Hopper.
a=out.index('import time as _lab_time\n');out=out[:a]+"if torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9:\n"+'\n'.join('    '+l for l in out[a:].splitlines())+'\n'
compile(out,'p5_early_history_board.py','exec');assert len(out.encode())<1_000_000
p=HERE/'out/p5_early_history_board.py';p.write_text(out);print('BOARD',hashlib.sha256(out.encode()).hexdigest(),len(out));print('PAYLOAD',hashlib.sha256(payload.encode()).hexdigest());print('BODY',hashlib.sha256(body.encode()).hexdigest())
