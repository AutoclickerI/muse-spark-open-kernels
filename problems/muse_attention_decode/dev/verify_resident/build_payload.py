"""One self-contained whole-layer resident-verify lab and native component host."""
import ast,base64,hashlib,lzma,re,sys
from pathlib import Path
import resgen
H=Path(__file__).resolve().parent;R=H.parents[3]
CONFIGS={'C2P':(2,True),'C4P':(4,True),'C8P':(8,True),'C16P':(16,True),'C4S':(4,False),'C8S':(8,False)}

def boot(src):
    sha=hashlib.sha256(src.encode()).hexdigest();blob=base64.b85encode(lzma.compress(src.encode(),preset=9)).decode()
    return 'import hashlib,base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\nassert hashlib.sha256(_BODY.encode()).hexdigest()=='+repr(sha)+'\n_FN='+repr('<verify-resident-'+sha[:12]+'>')+'\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n'

def main():
    (H/'out').mkdir(exist_ok=True)
    kernels={n:resgen.source(c,p) for n,(c,p) in CONFIGS.items()}
    runtime='KERNELS='+repr(kernels)+'\n'+(H/'runtime.py').read_text()
    base=(R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()=='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
    timeline=(R/'prof/lab/last_sliding_timeline.py').read_text();nodes=ast.parse(timeline).body
    header=['import gc,hashlib,json,linecache,math,statistics,time,types,base64,lzma','import torch','from torch.utils.cpp_extension import load_inline']
    for n in nodes:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('TL_ACTIVITY','TL_REPS') for t in n.targets):
            if getattr(n.targets[0],'id','')=='TL_ACTIVITY':
                activity=ast.literal_eval(n.value).replace('k->registersPerThread,k->streamId});','k->registersPerThread,k->streamId,k->blockY,k->blockZ,k->staticSharedMemory});')
                header.append('TL_ACTIVITY='+repr(activity))
            else:header.append(ast.get_source_segment(timeline,n))
        elif isinstance(n,ast.FunctionDef) and ((n.name.startswith('tl_') and n.name not in ('tl_host','tl_run','tl_prepare_selected')) or n.name in ('cs_copy','cs_delta','cs_profile')):
            text=ast.get_source_segment(timeline,n)
            if n.name=='cs_profile':
                text=text.replace("return graph,out,dict(inc=", "device_resources=list({(mapping[r[4]],name):dict(stage=mapping[r[4]],name=name.split('(')[0],grid=list(r[7:10]),block=[r[10],r[14],r[15]],dynamic_shared=r[11],static_shared=r[16],registers=r[12]) for name,r in records if r[4] in mapping and mapping[r[4]] in ('qkv_gate','attn')}.values())\n    return graph,out,dict(kernel_resources=device_resources,inc=")
            header.append(text)
    spec='import math,functools\nimport torch\n'
    for file in ('nvfp4.py','checks.py','reference.py'):
        src=(R/'problems/muse_decode_layer'/file).read_text()
        spec+='\n'.join(ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef)))+'\n'
    s=(H.parent/'early_history/lab_body.inc').read_text();checker=next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='eh_checker')
    header+=['PUBLIC_SPEC={"torch":torch}','exec('+repr(spec)+',PUBLIC_SPEC)',checker]
    body='\n'.join(header)+'\nBASE='+repr(base)+'\n'+runtime+'\ngenerate_weights=PUBLIC_SPEC["generate_weights"]\n'+(H/'lab_body.inc').read_text()
    compile(body,'resident_verify_body','exec');(H/'out/body.py').write_text(body)
    payload=boot(body)+'_VR_LAB=lab\ndef lab(deadline,weights,state=None):\n    return _VR_LAB(deadline,weights,state)\n';(H/'out/resident_verify_lab.py').write_text(payload)
    host=(R/'runs/muse_attention_decode/20261025T030917Z_submit_81cccbc19410433b.py').read_text();host=host[:host.rindex('if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:')]
    lines=host.splitlines(True)
    for n in reversed(ast.parse(host).body):
        if isinstance(n,ast.If) and '_lab' in ast.get_source_segment(host,n):lines[n.lineno-1:n.end_lineno]=[]
    host=''.join(lines)
    # CUDA literals are inserted after Python renaming, never text-substituted.
    cell=(H/'runtime.py').read_text()
    for name in ('KERNELS','_MODULES','_FREQ','_COUNTERS','TOPOLOGY','module','attention'):cell=re.sub(r'\b'+name+r'\b','vr_'+name,cell)
    host+='\nvr_KERNELS='+repr(kernels)+'\n'+cell+'''
_VR_BASE_FN=kernel_fn
def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg=data['config']
    if torch.cuda.get_device_capability()[0]==9 and cfg['q_len']==8 and cfg['window']==2048 and cfg['batch'] in (16,32) and cfg['num_q_heads']==8*cfg['num_kv_heads']:
        return vr_attention(data,name='C4P' if cfg['batch']==32 else 'C8P')
    return _VR_BASE_FN(data)
'''
    compile(host,'resident_component','exec');(H/'out/component.py').write_text(host)
    print('payload',hashlib.sha256(payload.encode()).hexdigest(),'component',hashlib.sha256(host.encode()).hexdigest(),'bytes',len(host));print('kernels',[(n,hashlib.sha256(s.encode()).hexdigest()[:8]) for n,s in kernels.items()])
if __name__=='__main__':main()
