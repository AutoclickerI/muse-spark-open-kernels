"""Reproducible sm90a production-loop SASS counts, with explicit issue-model limits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
OP=re.compile(r'/\*([0-9a-f]+)\*/\s+(?:(@!?[A-Z]+\d+)\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;')


def parse(text):
    labels={};ops=[];line=0;file='';pending=[]
    for textline in text.splitlines():
        m=re.search(r'//## File "([^"]+)", line (\d+)',textline)
        if m:file=m[1];line=int(m[2])
        m=re.match(r'(\.L_x_\d+):',textline.strip())
        if m:pending.append(m[1])
        m=OP.search(textline)
        if m:
            pc=int(m[1],16)
            for label in pending:labels[label]=pc
            pending=[]
            ops.append(dict(pc=pc,pred=m[2],op=m[3],arg=m[4],file=file,line=line))
    return ops,labels


def category(o):
    op=o['op'];arg=o['arg'];base=op.split('.')[0]
    if base in ('HMMA','HGMMA'):return 'mma'
    if base in ('HMUL2','HFMA2'):return 'scale'
    if base=='PRMT' and any(v in arg for v in ('0x4040','0x4141','0x4242','0x4343','0x1010','0x3232')):return 'scale'
    if '0x70007000' in arg or '0x7000' in arg and base in ('LEA','IADD3','IMAD'):return 'scale'
    if base in ('LOP3','ULOP3') and any(v in arg.lower() for v in ('0x81c081c0','0x60306030','0x9fcf9fcf')):return 'unpack'
    if base=='SHF' and '.W.' in op:return 'unpack'
    if base.startswith(('LD','ST')) or base in ('UBLKCP','UTMACMDF'):return 'load'
    if base in ('BRA','BSSY','BSYNC','BAR','WARPSYNC','WARPGROUP','SYNCS','FENCE','MEMBAR','NOP','YIELD','ISETP','PLOP3','R2P'):return 'control'
    return 'address'


def summarize(name,ops,weights,block_bytes):
    counts=Counter(category(o) for o in ops);groups=weights/16
    per={k:counts[k]/groups for k in ('unpack','scale','mma','load','address','control')}
    n=len(ops)/groups
    # User's budget:4TB/s corresponds to15B/SM/cycle. Four warp instructions
    # provide128 lane-ops/cycle; each per-lane group processes16 weights.
    bytes_per_cycle=128*block_bytes/n
    return dict(name=name,pc_begin=hex(ops[0]['pc']),pc_end=hex(ops[-1]['pc']),
        weights_per_lane=weights,static_instructions=len(ops),opcode_counts=dict(Counter(o['op'].split('.')[0] for o in ops)),
        per16=per,instructions_per16=n,lane_ops_per_weight=n/16,
        logical_block_bytes=block_bytes,ideal_four_warp_issue_TBs=4*bytes_per_cycle/15,
        caveat='Static issue ceiling, not measured throughput: ignores stalls, active-lane masks, functional-unit throughput, weight rereads and synchronization. LDS/LDGSTS counts include activation transport.')


def compile_one(prefix):
    p=HERE/(prefix+'.cu');cube=p.with_suffix('.cubin')
    result=subprocess.run(['nvcc','--cubin','-O3','-std=c++20','-arch=sm_90a','-lineinfo','-Xptxas=-v',str(p),'-o',str(cube)],capture_output=True,text=True)
    p.with_suffix('.ptxas.log').write_text(result.stdout+result.stderr);result.check_returncode()
    assert not re.search(r'[1-9]\d* bytes spill',result.stderr) and 'C7513' not in result.stderr,result.stderr
    sass=subprocess.run(['nvdisasm','-c','-g',str(cube)],capture_output=True,text=True,check=True).stdout
    p.with_suffix('.sass').write_text(sass)
    full=subprocess.run(['cuobjdump','--dump-sass',str(cube)],capture_output=True,text=True,check=True).stdout
    p.with_suffix('.cuobjdump.sass').write_text(full)
    assert not re.search(r'\b(?:LDL|STL)\b',full),'local-memory instruction in '+prefix
    resources=[int(x) for x in re.findall(r'Used (\d+) registers',result.stderr)]
    return sass,dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),registers=resources,no_spills=True)


def audit_one(prefix,sass):
    rows=[]
    for part in re.split(r'//-+ \.text\.',sass)[1:]:
        name=part.splitlines()[0].split(' ---',1)[0].strip()
        ops,labels=parse(part)
        if 'moe_fused' in name and prefix in ('norig','nlop'):
            mm=[i for i,o in enumerate(ops) if o['op'].startswith('HMMA.')]
            assert len(mm)==32,(prefix,len(mm))
            candidates=[i for i in range(max(0,mm[0]-80),mm[0]) if ops[i]['op'].startswith('LDS.128')]
            assert len(candidates)>=2,candidates
            # The first weight and scale LDS.128 start the fully unrolled K512
            # reader. Activation gathering precedes it and is audited separately.
            lo=candidates[0];rows.append(summarize(prefix+'_K512',ops[lo:mm[-1]+1],256,9))
            # Keep the concrete pre-loop activation gather for context; it is
            # route/segment dependent, so no fictitious per-weight normalization.
        if 'fused' in name and prefix in ('ibase','iword','idist'):
            candidates=[]
            for hi,o in enumerate(ops):
                m=re.search(r'`\((\.L_x_\d+)\)',o['arg'])
                if not o['op'].startswith('BRA') or m is None or m[1] not in labels:continue
                lo_pc=labels[m[1]]
                if lo_pc>=o['pc']:continue
                body=[v for v in ops if lo_pc<=v['pc']<=o['pc']]
                mm=[v for v in body if v['op'].startswith('HGMMA.')]
                # Eight WGMMA issues consume64 weights/lane: paired K64 stages
                # or one K128 stage. Outer scheduling loops contain more MMAs.
                if len(mm)!=8:continue
                widths={re.search(r'64x(\d+)x16',v['op'])[1] for v in mm}
                if len(widths)!=1:continue
                width=next(iter(widths));candidates.append((width,body))
            # An outer persistent-task back edge can contain the same static
            # eight MMAs as the inner K loop. Keep the smallest natural loop
            # for each width in this entry, never normalize epilogues as K work.
            for width in sorted({w for w,b in candidates},key=int):
                body=min((b for w,b in candidates if w==width),key=len)
                label=prefix+'_'+('W13' if 'ELb1EE' in name or 'ELb1ELi1' in name else 'W2')+'_N'+width
                row=summarize(label,body,64,9 if prefix=='ibase' else 10);row['entry']=name;rows.append(row)
    return rows


def main():
    result={};rows=[]
    for prefix in ('norig','nlop','ibase','iword','idist','ntrace'):
        sass,res=compile_one(prefix);result[prefix]=res
        rows+=audit_one(prefix,sass)
    assert any(r['name']=='norig_K512' for r in rows) and any(r['name']=='nlop_K512' for r in rows)
    assert any(r['name'].startswith('ibase_') for r in rows) and any(r['name'].startswith('iword_') for r in rows)
    payload=dict(resources=result,loops=rows,budget=dict(reference_TBs=4.,bytes_per_SM_cycle=15.,warp_issue_per_SM_cycle=4,lane_ops_per_SM_cycle=128),
        conclusion='Four-warp aggregate issue alone is an upper bound; phase and paired H200 data must identify the binding limit. One-LOP3 blend and word scales reduce actual SASS; no H200 gain is inferred from local timing.')
    (HERE/'audit.json').write_text(json.dumps(payload,indent=2)+'\n')
    for r in rows:print(r['name'],r['instructions_per16'],r['per16'],f'issue_roof={r["ideal_four_warp_issue_TBs"]:.2f}TB/s')
    print('ISSUE_NATIVE_PASS',sum(len(v['registers']) for v in result.values()),'entries including route/finalizer;',len(rows),'loops')


if __name__=='__main__':main()
