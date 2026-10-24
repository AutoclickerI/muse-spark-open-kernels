"""Compare saved attention profiles offline; never requests or waits for a judge result."""
import argparse
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
DEFAULT=ROOT/'problems/muse_attention_decode/profiles/cand_attn_verify_stream.d98ae45d7204438d.json'


def shape(case):
    return case['batch'],case.get('q_len',1),case['layer']


def report_parts(report):
    kernels=[]
    for line in report.splitlines():
        match=re.match(r'^(.*?)\s+([0-9.]+) µs\s+([0-9.]+)%\s+(\d+)\s*$',line)
        if match:
            kernels.append(dict(name=match[1].strip(),us=float(match[2]),calls=int(match[4])))
    total=re.search(r'^total\s+([0-9.]+) µs',report,re.M)
    return dict(total_us=float(total[1]) if total else None,kernels=kernels)


def compare(candidate,control):
    candidate=json.loads(candidate.read_text());control=json.loads(control.read_text())
    cb={shape(r['case']):r for r in candidate['benchmarks']}
    ob={shape(r['case']):r for r in control['benchmarks']}
    cp={shape(r['case']):report_parts(r['report']) for r in candidate['profiles']}
    op={shape(r['case']):report_parts(r['report']) for r in control['profiles']}
    rows=[]
    for key,r in cb.items():
        old=ob[key];b,q,layer=key
        row=dict(batch=b,q_len=q,layer=layer,load=r['case']['load'],
                 candidate_call_us=r['median_ns']/1000,control_call_us=old['median_ns']/1000,
                 candidate_profile=cp[key],control_profile=op[key],gate=r['status'])
        row['candidate_main_us']=sum(k['us'] for k in cp[key]['kernels'] if 'verify8_main' in k['name'])
        row['candidate_combine_us']=sum(k['us'] for k in cp[key]['kernels'] if 'verify8_combine' in k['name'])
        rows.append(row)
    return dict(candidate_job=candidate['job_id'],control_job=control['job_id'],rows=rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('candidate',type=Path)
    ap.add_argument('--control',type=Path,default=DEFAULT)
    ap.add_argument('--json',type=Path)
    args=ap.parse_args()
    result=compare(args.candidate,args.control)
    if args.json:args.json.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['rows']:
        print(f"b{r['batch']} q{r['q_len']} {r['layer']}: call {r['candidate_call_us']:.2f}/{r['control_call_us']:.2f} "
              f"profile {r['candidate_profile']['total_us']}/{r['control_profile']['total_us']} "
              f"producer={r['candidate_main_us']:.1f} combine={r['candidate_combine_us']:.1f} {r['gate']}")
