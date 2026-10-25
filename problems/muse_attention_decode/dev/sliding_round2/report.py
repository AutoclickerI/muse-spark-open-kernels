"""Read the round2 receipt archive; show every paired arm and its critical-path displacement."""
import argparse,base64,json,lzma,math,statistics
from pathlib import Path

def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from strings(v)
    elif isinstance(x,list):
        for v in x:yield from strings(v)

def stats(v):return statistics.mean(v),statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else float('inf')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
    raw=a.receipt.read_text()
    try:chunks=list(strings(json.loads(raw)))
    except json.JSONDecodeError:chunks=[raw]
    blobs={line.split('R2_XZ85 ',1)[1].strip() for chunk in chunks for line in chunk.splitlines() if 'R2_XZ85 ' in line}
    assert len(blobs)==1,'expected one complete round2 archive; collect/read the judge receipt first'
    rows=json.loads(lzma.decompress(base64.b85decode(blobs.pop())))
    for row in rows:
        assert all(v==row['seeds'] for v in row['task_passed'].values()),row['case']
        times=dict(zip(row['names'],row['times']));assert all(len(v)==row['seeds'] for v in times.values())
        base=times['A'];noise=[x/y for x,y in zip(times['A2'],base)];nm,ne=stats(noise)
        print(f"{row['case']} seeds={row['seeds']} rounds={row['rounds']} A:A={nm:.6f} +/-{ne:.6f}")
        candidates=[]
        for name in row['arms']:
            ratio=[x/y for x,y in zip(times[name],base)];m,se=stats(ratio)
            to_noise=[x/y for x,y in zip(times[name],times['A2'])];mn,en=stats(to_noise)
            clean=1-m>abs(nm-1)+3*se and mn+3*en<1
            inc=row['critical_ns'][name];old=row['critical_ns']['A'];order=row['critical_order']
            shift={k:round((inc[i]-old[i])/1000,3) for i,k in enumerate(order)}
            print(f"  {name:12s} {m:.6f} +/-{se:.6f} vsA2={mn:.6f} +/-{en:.6f} delta_us={shift} {'CONFIRM' if clean else 'hold'}")
            if clean:candidates.append((m,name,row['arms'][name]))
        if candidates:print('  first confirmation:',min(candidates))
    print('CONFIRM ranks a follow-up; integration still owns paired six-load promotion.')
if __name__=='__main__':main()
