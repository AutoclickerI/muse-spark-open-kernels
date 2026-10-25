"""Compile the rebased NF hint's two actual generated producers for sm90a."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval,patches


def main():
    source,meta=patches.compose(['boundary_P']);p=H/'generated/variants/boundary_P.py';p.write_text(source)
    m=local_eval.load_module('submission',p);ns=m._nf_vendor();units=[]
    def capture(name,cpp_sources='',cuda_sources='',**kw):
        if cuda_sources:units.append((name,cuda_sources))
        return None
    ns['load_inline']=capture;ns['_nf_arch_flag']=lambda:'-gencode=arch=compute_90a,code=sm_90a'
    family,cfg,mt,st,red,ntmax,extra=m.NF_SMALL[16,'unpooler']
    inst=(1,mt,1,st,red,0,ntmax,0,8192,4096,*cfg,*extra)
    ns['dd_decodediet_cps_build']('sm',[inst],2)
    native=ns['dd_decodediet_tt_mw_cuda_source']([(32,64,4,2,2,True,1),(64,64,4,2,2,True,1)])
    units.append(('unpool32_hint',m._port_cuda(native,m32w=True)))
    rows=[];out=H/'generated/native_hint';out.mkdir(exist_ok=True)
    for name,code in units:
        assert 'prefetch.global.L2' in code
        first=code.index('s16_w=');wait=code.index('griddepcontrol.wait;',first)
        assert first<wait
        path=out/(name+'.cu');path.write_text(code)
        r=subprocess.run(['nvcc','-std=c++20','-O3','-arch=sm_90a','--cubin',str(path),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True)
        row=dict(name=name,sha256=hashlib.sha256(code.encode()).hexdigest(),ok=r.returncode==0,diagnostic=(r.stdout+r.stderr)[-1500:] if r.returncode else '')
        rows.append(row);print(row,flush=True)
    (H/'native_hint_audit.json').write_text(json.dumps(dict(candidate_sha256=meta['sha256'],rows=rows,passed=all(r['ok'] for r in rows)),indent=2)+'\n')
    assert len(rows)==2 and all(r['ok'] for r in rows)


if __name__=='__main__':main()
