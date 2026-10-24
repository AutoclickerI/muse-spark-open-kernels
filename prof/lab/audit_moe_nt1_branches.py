"""Compile every ready small-M branch and normalize the actual Hopper decoder SASS."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/screen/nt1_branches'


def main():
    spec=importlib.util.spec_from_file_location('branch_audit',ROOT/'prof/lab/moe_nt1_branches.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    OUT.mkdir(parents=True,exist_ok=True);result={}
    for kind,source in m.NT1_SOURCES.items():
        path=OUT/(kind+'.cu');path.write_text(source);cubin=path.with_suffix('.cubin')
        r=subprocess.run(['nvcc','-O3','-std=c++20','-lineinfo','-arch=sm_90a','-Xptxas=-v',
                          '-cubin',str(path),'-o',str(cubin)],capture_output=True,text=True)
        path.with_suffix('.build.log').write_text(r.stdout+r.stderr)
        assert not r.returncode,r.stderr
        sass=subprocess.check_output(['nvdisasm','-g',str(cubin)],text=True)
        path.with_suffix('.sass').write_text(sass)
        lines=source.splitlines();counts=dict(extract=0,rotate=0,multiply=0,scale=0,mma=0)
        line=0
        dq_a=next(i+1 for i,s in enumerate(lines) if 'void dq8(' in s)
        dq_b=next(i+1 for i,s in enumerate(lines) if 'struct PrefetchState' in s)
        sf_a=next(i+1 for i,s in enumerate(lines) if 'const uint32_t Sr =' in s)
        sf_b=next(i+1 for i,s in enumerate(lines) if 'uint32_t o[16]' in s)
        for s in sass.splitlines():
            loc=re.search(r'//## File .*?, line (\d+)',s)
            if loc:line=int(loc[1])
            op=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',s)
            if not op:continue
            full_op=op[1];op=full_op.split('.')[0]
            if dq_a<=line<dq_b:
                if op=='LOP3':counts['extract']+=1
            # The funnel-shift intrinsic is attributed to its CUDA header,
            # so count the wrapping rotate opcode rather than a .cu line.
            if full_op.startswith('SHF.R.W.'):counts['rotate']+=1
            if sf_a<=line<sf_b and op in ('PRMT','LEA'):counts['scale']+=1
            if op in ('HMUL2','HFMA2'):counts['multiply']+=1
            if op=='HMMA':counts['mma']+=1
        result[kind]=dict(registers=[int(v) for v in re.findall(r'Used (\d+) registers',r.stderr)],
            spills=[int(v) for v in re.findall(r'(\d+) bytes spill',r.stderr)],
            local_instructions=len(re.findall(r'\b(?:LDL|STL)\b',sass)),lane_tile=counts,
            decoder_per16=(sum(counts[k] for k in ('extract','rotate','multiply','scale'))/16
                           if kind!='raw' else None))
        print(kind,result[kind],flush=True)
    for kind,expected in [('recode',27),('lop26',26)]:
        row=result[kind];assert row['decoder_per16']==expected,row
        assert row['registers'][0]<=124 and not any(row['spills']),row
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
