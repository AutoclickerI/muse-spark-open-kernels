"""Build one round-2 candidate and its complete paired whole-layer payload."""
import ast,base64,hashlib,lzma,re,sys
from pathlib import Path
import kernels
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import h200_lab

def functions(path,names):
    s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)

def runtime():
    src=(HERE/'runtime.py').read_text()
    return 'KERNELS='+repr({k:kernels.source(k) for k in ('last','early','dsm')})+'\n'+src

def boot(src):
    tag=hashlib.sha256(src.encode()).hexdigest();blob=base64.b85encode(lzma.compress(src.encode(),preset=9)).decode()
    return ('import hashlib,base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n'
    'assert hashlib.sha256(_BODY.encode()).hexdigest()=='+repr(tag)+'\n_FNAME='+repr('<p5r2-'+tag[:12]+'>')+'\n'
    'linecache.cache[_FNAME]=(len(_BODY),None,_BODY.splitlines(True),_FNAME)\nexec(compile(_BODY,_FNAME,"exec"),globals())\n')

def main():
    (HERE/'out').mkdir(exist_ok=True)
    base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()=='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
    # Rebuild from tracked inputs; generated reset out/ files are not build
    # dependencies. Freeze public checker math, never import it on the judge.
    timeline=(ROOT/'prof/lab/last_sliding_timeline.py').read_text();nodes=ast.parse(timeline).body
    header=['import gc,hashlib,json,linecache,math,statistics,time,types,base64,lzma',
            'import torch','from torch.utils.cpp_extension import load_inline']
    for n in nodes:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('TL_ACTIVITY','TL_REPS') for t in n.targets):header.append(ast.get_source_segment(timeline,n))
        elif isinstance(n,ast.FunctionDef) and ((n.name.startswith('tl_') and n.name not in ('tl_host','tl_run','tl_prepare_selected')) or n.name in ('cs_copy','cs_delta','cs_profile')):header.append(ast.get_source_segment(timeline,n))
    spec='import math,functools\nimport torch\n'
    for name in ('nvfp4.py','checks.py','reference.py'):
        text=(ROOT/'problems/muse_decode_layer'/name).read_text()
        spec+='\n'.join(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))+'\n'
    header+=['PUBLIC_SPEC={"torch":torch}','exec('+repr(spec)+',PUBLIC_SPEC)',functions(HERE.parent/'early_history/lab_body.inc',('eh_checker',))]
    body='\n'.join(header)+'\nBASE='+repr(base)+'\n'+runtime()
    old_dsm=kernels.build.source()
    assert hashlib.sha256(old_dsm.encode()).hexdigest()=='e43cb5134924fff5e3f246759922b557259291e830804892a67697ee27381ca1'
    body+='\nKERNELS["dsm_old"]='+repr(old_dsm)+'\n'
    # Public weight generator is already included in the fixed checker namespace.
    body+='generate_weights=PUBLIC_SPEC["generate_weights"]\n'+(HERE/'lab_body.inc').read_text()
    compile(body,'sliding_round2_payload','exec');(HERE/'out/body.py').write_text(body)
    packet=boot(body);(HERE/'out/sliding_round2_lab.py').write_text(packet)
    # Frozen attention incumbent, with its import-only search removed.
    host=(ROOT/'runs/muse_attention_decode/20261025T030917Z_submit_81cccbc19410433b.py').read_text()
    host=host[:host.rindex('if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:')]
    lines=host.splitlines(True)
    for n in reversed(ast.parse(host).body):
        if isinstance(n,ast.If) and '_lab' in ast.get_source_segment(host,n):lines[n.lineno-1:n.end_lineno]=[]
    host=''.join(lines)+'\n'
    cell=runtime();assigns=ast.parse(cell).body;lines=cell.splitlines(True);kernel_literal=None
    for n in reversed(assigns):
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='KERNELS':kernel_literal=ast.literal_eval(n.value);lines[n.lineno-1:n.end_lineno]=[]
    cell=''.join(lines)
    for name in ('KERNELS','_MODULES','_COUNTERS','_FREQ','module','attention','ARMS','SELECTED','apply_arm'):cell=re.sub(r'\b'+name+r'\b','r2_'+name,cell)
    host+='r2_KERNELS='+repr(kernel_literal)+'\n'+cell+'''
_R2_BASE_FN=kernel_fn
def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg=data['config'];key=cfg['batch'],cfg['q_len']
    if key in r2_SELECTED and cfg['window']==2048 and cfg['num_q_heads']==8*cfg['num_kv_heads']:
        return r2_apply_arm(data,None,r2_SELECTED[key])
    return _R2_BASE_FN(data)
'''
    (HERE/'out/component.py').write_text(host)
    tag=hashlib.sha256(packet.encode()).hexdigest()[:16]
    out=host+'\n'+h200_lab.WRAPPER.format(names='sliding_round2 '+tag,base='attention portfolio + P5 second-round one-launch rows; paired lab exactS14',payloads=[('sliding_round2',tag,packet)],budget=650,limit=15000,print_line='print(line, flush=True)')
    out+='\nif torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9:\n'+'\n'.join('    '+l for l in h200_lab.IMPORT_WRAPPER.format(budget=650).splitlines())+'\n'
    compile(out,'sliding_round2_board','exec');assert len(out.encode())<1_000_000
    p=HERE/'out/sliding_round2_board.py';p.write_text(out)
    print('BOARD',hashlib.sha256(out.encode()).hexdigest(),'bytes',len(out.encode()));print('BODY',hashlib.sha256(body.encode()).hexdigest())
if __name__=='__main__':main()
