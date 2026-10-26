"""Compile all native arms and retain line-attributed instruction evidence."""
import collections,concurrent.futures,hashlib,json,re,subprocess,sys
from pathlib import Path
import kernels_new as k
H=Path(__file__).resolve().parent;(H/'out').mkdir(exist_ok=True)
def compile_one(name):
    f,a=name[0],name[1:];s=k.oracle(f) if a=='M' else k.source(f,a)
    p=H/'out'/(name+'.cu');p.write_text(s)
    r=subprocess.run(['nvcc','-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-cubin',str(p),'-o',str(p.with_suffix('.cubin'))],capture_output=True,text=True)
    log=r.stdout+r.stderr;p.with_suffix('.compile.log').write_text(log);assert r.returncode==0,(name,log[-5000:]);assert not re.search(r'C75\d+',log),(name,log)
    sass=subprocess.check_output(['nvdisasm','--print-line-info',str(p.with_suffix('.cubin'))],text=True);p.with_suffix('.sass').write_text(sass)
    resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.cubin'))],text=True)
    ops=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_.]+)',sass))
    regions={};lines=s.splitlines()
    anchors=['DI void prep_decode','DI void prep_verify','auto issue=','DI void input_pairs','DI float2 rope_cs','__global__ __launch_bounds__']
    for anchor in anchors:
        hits=[i+1 for i,l in enumerate(lines) if anchor in l]
        regions[anchor]=hits
    # nvdisasm retains exact line annotations and SASS in ignored out; the
    # compact record pins bytes, opcodes, resources and selected SASS excerpts.
    excerpts=[l.strip() for l in sass.splitlines() if re.search(r'\b(LDGSTS|QGMMA|MUFU|F2FP|SHFL|SYNCS|UTMA|TMA)\b',l)][:32]
    row=dict(name=name,sha256=hashlib.sha256(s.encode()).hexdigest(),resources=resources,opcodes=dict(ops),anchors=regions,excerpts=excerpts,
             copy_sites=len(re.findall(r'\bLDGSTS\b',sass)),tensor_sites=len(re.findall(r'\bQGMMA\b',sass)),TMA_sites=len(re.findall(r'\b(?:UTMA|TMA)\b',sass)),no_C75=True)
    assert row['copy_sites'] and row['tensor_sites'];print('PASS',name,resources.replace('\n',' ')[-230:],flush=True);return row
if __name__=='__main__':
    hoist='--hoist' in sys.argv
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(compile_one,('dA','dH','vA','vH') if hoist else ('dA','dQ','dM','vA','vQ','vT','vM')))
    (H/('native_hoist.json' if hoist else 'native.json')).write_text(json.dumps(rows,indent=2)+'\n')
