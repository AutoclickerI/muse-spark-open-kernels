"""Read projection receipts and optionally wait on filesystem completion events (no polling)."""
import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct

ROOT=Path(__file__).resolve().parents[3]
RECEIPTS=ROOT/'runs/muse_bf16_proj'
PENDING=ROOT/'runs/pending'


def digest(path):
    path=path.resolve()
    job=json.loads(path.read_text())
    result=job.get('result') or {}
    source=path.with_suffix('.py')
    text=result.get('log_tail','')
    log=job.get('output',{}).get('log')
    if log and Path(log).exists(): text=Path(log).read_text()
    choices={};tops={};tune={}
    for line in text.splitlines():
        match=re.match(r'(?:PORTFOLIO(?:_V2)?|REDUCE(?:_FAST|_PACKED|_HIER)?|FUSED(?:_LEAN|_SELECT)?) (\w+) m=(\d+) incumbent_us=([\d.]+) chosen=(.*?) chosen_us=([\d.]+) top=(.*)',line)
        if not match: continue
        layer,m,baseline,selected,chosen,top=match.groups()
        key=f'{layer}:{m}'
        choices[key]=ast.literal_eval(selected)
        tops[key]=ast.literal_eval(top)
        tune[key]={'incumbent_us':float(baseline),'chosen_us':float(chosen)}
    shapes=[]
    for row in result.get('records',[]):
        if 'speedup' not in row: continue
        c=row['case']
        shapes.append({'layer':c['layer'],'m':c['m'],'sglang_us':row['baseline_ns']/1000,
                       'ours_us':row['median_ns']/1000,'speedup':row['speedup']})
    return {'receipt':str(path.relative_to(ROOT)),'job':job.get('id',path.stem.split('_')[-1]),
            'candidate':Path(log).name.split('.')[0] if log else None,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
            'passed':result.get('passed'),'status':result.get('status'),'speedup':result.get('speedup'),
            'flags':result.get('flags'),'error':result.get('error'),
            'choices':choices,'top_variants':tops,'prepare_times':tune,'shapes':shapes}


def watch():
    seen={p.name for p in RECEIPTS.glob('*.json')}
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(0)
    if fd<0: raise OSError(ctypes.get_errno(),'inotify_init1')
    watches={}
    try:
        for folder in (RECEIPTS,PENDING):
            wd=libc.inotify_add_watch(fd,os.fsencode(folder),0x8|0x80)
            if wd<0: raise OSError(ctypes.get_errno(),f'inotify_add_watch {folder}')
            watches[wd]=folder
        print(f'WATCH ready: {len(seen)} completed receipts; waiting on filesystem events',flush=True)
        while True:
            block=os.read(fd,65536); offset=0
            while offset<len(block):
                wd,mask,cookie,length=struct.unpack_from('iIII',block,offset)
                name=block[offset+16:offset+16+length].split(b'\0',1)[0].decode()
                offset+=16+length
                path=watches[wd]/name
                if not name.endswith('.json') or not path.exists(): continue
                if watches[wd]==RECEIPTS:
                    if name in seen: continue
                    data=digest(path);seen.add(name)
                    print('RECEIPT '+json.dumps(data),flush=True)
                else:
                    job=json.loads(path.read_text())
                    if job.get('problem')=='muse_bf16_proj':
                        print('PENDING '+json.dumps({'job':path.stem,'file':job.get('file'),'mode':job.get('mode')}),flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        os.close(fd)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('receipts',nargs='*',type=Path);ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    if args.watch: watch()
    else:
        for path in args.receipts or sorted(RECEIPTS.glob('*.json')): print(json.dumps(digest(path)))
