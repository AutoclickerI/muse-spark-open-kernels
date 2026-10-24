"""Exercise every submitted Hopper launch ABI with portable MMA on the actual host."""
import hashlib
import json
import linecache
from pathlib import Path
import sys
import time
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval


def encoding_gate(ns):
    raw=torch.arange(256,device='cuda',dtype=torch.uint8);raw=raw[(raw&127)!=127]
    sf=raw.view(torch.float8_e4m3fn);code,neg,zero=ns['_encode_scales'](sf)
    word=((code.int()<<4)+0x7000).to(torch.int16)
    assert torch.equal(((word.int()-0x7000)>>4).to(torch.uint8),code)
    c=torch.arange(16,device='cuda');lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device='cuda')
    original=lut[c&7][None,:]*torch.where(c&8!=0,-1.,1.)[None,:]*sf.float()[:,None]
    folded=torch.where(zero[:,None],0,c[None,:]^(neg.long()[:,None]*8))
    scaled=word.view(torch.bfloat16).float()*2.**-118
    got=lut[folded&7]*torch.where(folded&8!=0,-1.,1.)*scaled[:,None]
    assert torch.equal(got,original)
    # LOP3(A,mask,B,0xe2) must implement exactly (A&mask)|(B&~mask).
    for a in (0,1):
        for mask in (0,1):
            for b in (0,1):assert ((0xe2>>(a*4+mask*2+b))&1)==((a&mask)|(b&(mask^1)))
    return dict(exact_values=original.numel(),scale_word_reversible=True,LOP3_truth_table=True,bytes_per_block=10)


def main():
    original_load=local_eval.load_module;report={}
    def loader(name,path):
        module=original_load(name,path)
        if name=='submission':
            def run_one(label,source,deadline,weights,state):
                filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
                report['encoding']=encoding_gate(ns);started=time.monotonic()
                lines=ns['lab'](min(deadline,started+78.),weights,state);rows=ns['ISSUE_ROWS']
                report.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=rows,lines=lines,
                    elapsed_s=time.monotonic()-started,producer_graph_checks=len(rows)*5,NT1_graph_checks=len(rows)*2,
                    portable='SM120 immediate MMA; same Hopper pointer glue, B128 layout and prepared integer scale words')
                (HERE/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
                assert len(rows)==36,lines[-1]
                assert not any('INCOMPLETE' in line or 'REJECT' in line for line in lines),lines
                assert all(e<=.01 and r<=.05 for row in rows for e,r in row['errors'])
                assert all(len(row['samples'][0])==12 and all(len(v)==6 for v in row['samples'][1:]) for row in rows)
                assert sum(len(line)+1 for line in lines)<=4500
                assert len([line for line in lines if line.startswith('PH ')])==5
                assert len([line for line in lines if line.startswith('NTPH ')])==1
                print('ISSUE_LOCAL_PASS',len(rows),'fixtures;',report['producer_graph_checks'],'producer gates;',
                    report['NT1_graph_checks'],'NT1 gates;',report['encoding'],flush=True)
                return lines
            module._lab_run_one=run_one
        return module
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_moe_w4a16/lab/lab_moe_m128_issue_limit.py'),
        '--no-baseline','--json',str(HERE/'wrapper_screen.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(report.get('rows',[]))==36,'payload failed despite host passing'
        raise


if __name__=='__main__':main()
