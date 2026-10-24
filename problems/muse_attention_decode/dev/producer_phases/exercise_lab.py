"""Exercise the full lab control flow with portable FA3; check deadline and output budget."""
import hashlib,inspect,json,sys,time
from pathlib import Path
from portable import portable_fa3
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
from local_eval import load_module
result={}
for name in ('verify8_producer_phases','verify1_sliding_fa3_pair'):
    p=ROOT/'prof/lab'/f'{name}.py';m=load_module(name+'_exercise',p)
    source=inspect.getsource(m._attention_fa3_public).replace('    from sgl_kernel.flash_attn import flash_attn_with_kvcache\n','')
    m.flash_attn_with_kvcache=portable_fa3
    exec(compile(source,'<portable_public_fa3>','exec'),m.__dict__)
    m.PDL=False
    start=time.monotonic();lines=m.lab(start+80,{})
    elapsed=time.monotonic()-start
    assert not any('gate=F' in line or 'GATE_ERROR' in line for line in lines),lines
    for line in lines:
        if line.startswith('ROW {'):assert json.loads(line[4:])['gate'],line
    assert elapsed<80,elapsed
    assert sum(len(line)+1 for line in lines)<4500,len('\n'.join(lines))
    result[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),elapsed_s=elapsed,chars=sum(len(line)+1 for line in lines),lines=lines,fa3='portable operation; exact layer Python/Triton glue, PDL disabled locally')
    print(name,elapsed,flush=True)
    print('\n'.join(lines),flush=True)
(ROOT/'problems/muse_attention_decode/dev/producer_phases/deadline_local.json').write_text(json.dumps(result,indent=2)+'\n')
