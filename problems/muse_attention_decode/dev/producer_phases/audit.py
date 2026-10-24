"""Rebuild sm90a code and preserve resource and opcode evidence, without a GPU."""
import ast,collections,hashlib,json,re,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def opcounts(sass):
    rows=[]
    for part in sass.split('Function : ')[1:]:
        name=part.splitlines()[0]
        ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]+)',part)
        count=collections.Counter(ops)
        rows.append(dict(symbol=name,instructions=len(ops),static_code_bytes=16*len(ops),opcodes=dict(count),
            local_loads=sum(v for k,v in count.items() if k.startswith('LDL')),
            local_stores=sum(v for k,v in count.items() if k.startswith('STL'))))
    return rows


def main():
    original=ROOT/'problems/muse_attention_decode/dev/flashdecode/verify8.cu'
    archive=ROOT/'runs/muse_attention_decode/20261024T154806Z_submit_be19da94dc534323.py'
    tree=ast.parse(archive.read_text())
    frozen=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VERIFY8_CUDA')
    assert frozen==original.read_text()
    version=subprocess.run(['nvcc','--version'],text=True,capture_output=True,check=True).stdout
    rows=[]
    jobs=[('baseline',original,'90a'),('baseline90',original,'90'),('instrumented',HERE/'instrumented.cu','90a'),('widened',HERE/'widened.cu','90a'),('mma_probe90',HERE/'mma8_probe.cu','90a'),('mma_probe120',HERE/'mma8_probe.cu','120')]
    with tempfile.TemporaryDirectory() as td:
        for name,src,arch in jobs:
            cubin=Path(td)/(name+'.cubin')
            cmd=['nvcc','-O3','-std=c++17','-arch=sm_'+arch,'-lineinfo','--cubin','--ptxas-options=-v,--warn-on-spills',str(src),'-o',str(cubin)]
            p=subprocess.run(cmd,capture_output=True,text=True,check=True)
            sass=subprocess.run(['cuobjdump','--dump-sass',str(cubin)],capture_output=True,text=True,check=True).stdout
            resource=subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],capture_output=True,text=True,check=True).stdout
            statics=opcounts(sass)
            rows.append(dict(name=name,source=str(src.relative_to(ROOT)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),arch='sm_'+arch,
                command=cmd[:-2]+['<cubin>'],ptxas=p.stdout+p.stderr,resources=resource,functions=statics))
            if name.startswith('mma_probe'):
                (HERE/(name+'.sass')).write_text(sass)
            if name=='baseline':
                # Preserve the hot loop's conversion/HMMA sequence; full output is reproducible.
                lines=sass.splitlines();idx=next(i for i,line in enumerate(lines) if 'HMMA.' in line)
                (HERE/'baseline_mma_excerpt.sass').write_text('\n'.join(lines[max(0,idx-45):idx+28])+'\n')
            print(name,resource.strip(),flush=True)
    evidence=dict(compiler=version,frozen_receipt='be19da94dc534323',source_matches_archive=True,rows=rows)
    (HERE/'sass_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('wrote sass_evidence.json',flush=True)

if __name__=='__main__':main()
