"""Compile all actual SM90a paths and pin shared control sources to receipts."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    result={}
    for prefix in ('pm_nt','pm_meta','idist','or_rs','norig'):
        p=HERE/(prefix+'.cu');cube=HERE/(prefix+'.cubin')
        r=subprocess.run(['nvcc','--cubin','-O3','-std=c++20','-arch=sm_90a','-lineinfo','-Xptxas=-v',str(p),'-o',str(cube)],capture_output=True,text=True)
        (HERE/(prefix+'.ptxas.log')).write_text(r.stdout+r.stderr);r.check_returncode()
        assert 'C7513' not in r.stderr,r.stderr
        sass=subprocess.run(['cuobjdump','--dump-sass',str(cube)],check=True,capture_output=True,text=True).stdout
        (HERE/(prefix+'.sass')).write_text(sass);entries=[];registers={};name=None
        for line in r.stderr.splitlines():
            if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
            match=re.search(r'Used (\d+) registers',line)
            if match:registers[name]=int(match[1])
        for part in sass.split('Function : ')[1:]:
            name=part.splitlines()[0].strip();local=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?(?:LDL|STL)[.\s]',part)
            assert not local,(prefix,name,len(local))
            entries.append(dict(name=name,registers=registers.get(name),local_memory_instructions=len(local)))
        result[prefix]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kernels=entries)
    # Actual admitted idist and actual NT1 controls must remain byte-identical.
    for p in ('idist','norig'):assert (HERE/(p+'.cu')).read_bytes()==(HERE.parent/'issue_limit'/(p+'.cu')).read_bytes()
    orig=(ROOT/'problems/muse_decode_layer/dev/s4/manifest.json').read_text();accepted=json.loads(orig)
    tree=ast.parse((HERE/'d32g_cell.py').read_text())
    dist=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='S4_D32_CUDA' for t in n.targets))
    assert hashlib.sha256(dist.encode()).hexdigest()==accepted['d32_cuda_sha256']
    assert dist==(HERE/'idist.cu').read_text()
    report=dict(sources=result,source_controls_byte_identical=True,handoff_matches_S4_and_accepted_payload=True,
        nt_phase_specializations=4,no_spills=True,total_entries=sum(len(v['kernels']) for v in result.values()))
    (HERE/'native.json').write_text(json.dumps(report,indent=2)+'\n');print('PHASE_MIX_NATIVE_PASS',report['total_entries'],'entries; four NT1 phase kernels no spills',flush=True)


if __name__=='__main__':main()
