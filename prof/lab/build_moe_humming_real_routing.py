#!/usr/bin/env python3
"""Finish Humming/reference coverage using a compact loaded-route export from the exact lab."""
import argparse
import hashlib
import re
from pathlib import Path
from build_moe_real_routing import ROOT,HERE,namespace,build
from read_moe_real_routing import read

FAMILIES=('2','4','5','6','6k','7','8')


def make(job=None,output=None):
    rec=read(job) if job else None
    if rec:
        if not rec['loaded'] or not rec['verify_corrected']:
            raise ValueError('corrected exact loaded-router receipt required')
        match=re.search(r'^PACK_R ms=8,16,32,128,256 first_fixture zlib_u8=(\S+)',rec['log'],re.M)
        if not match:raise ValueError('receipt has no routing export')
        packed=match[1];route_job=rec['job']
        previous=sorted({min((r['us'],r['kernel']) for r in rec['rows']
                            if r['m']==m and r['routing']=='R' and 'ref' not in r['kernel'])[1]
                         for m in (8,16,32,128,256)})
    else:
        # This output is for local ABI screening only and cannot execute its Hopper branch.
        packed='';route_job='LOCAL_SCREEN_ONLY';previous=['v9']
    build()
    parts=[(HERE/'moe_real_routing.py').read_text()]
    calls=[]
    for suffix in FAMILIES:
        p=ROOT/f'problems/muse_moe_w4a16/cand_humming{suffix}.py'
        s=p.read_text()
        s=re.sub(r'self\.compile\(range\(1, PRECOMPILED_MAX_BS\s*\+\s*1\)\)',
                 'self.compile((8,16,32,128,256))',s)
        prefix='h'+suffix+'_'
        parts += [f'# {p.relative_to(ROOT)} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}\n',namespace(s,prefix)]
        calls.append(f"('h{suffix}',{prefix!r},{prefix}prepare,{prefix}kernel_fn)")
    parts += ['PROBE_HUMMING=('+','.join(calls)+',)\n',
              f'PROBE_PACK_R={packed!r}\nPROBE_ROUTE_JOB={route_job!r}\nPROBE_PREVIOUS={previous!r}\n',
              (HERE/'moe_humming_real_routing_driver.py').read_text()]
    target=Path(output) if output else HERE/'moe_humming_real_routing.py'
    if not rec and output is None:raise ValueError('local-only build needs an explicit output')
    target.write_text('\n'.join(parts));print(target,len(target.read_bytes()),hashlib.sha256(target.read_bytes()).hexdigest())
    return target

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('job',nargs='?');a.add_argument('-o',type=Path);args=a.parse_args();make(args.job,args.o)
