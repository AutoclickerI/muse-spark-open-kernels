"""Low-register follow-up: issue first weight cachelines before wait, keep the original tile pipeline."""
import ast
import hashlib
from pathlib import Path
from patch import patch,once

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def patch_hint(base):
    fixed=patch(base);original={n.name:ast.get_source_segment(base,n) for n in ast.parse(base).body if isinstance(n,ast.FunctionDef)}
    changes=[];lines=fixed.splitlines(True)
    for n in ast.parse(fixed).body:
        if not isinstance(n,ast.FunctionDef) or n.name not in ('_gemv_kernel','_gb_kernel','_gi_kernel'):continue
        code=original[n.name]
        if n.name!='_gemv_kernel':
            wait='    tl.extra.cuda.gdc_wait()' if n.name=='_gi_kernel' else '    _pdl_wait(USE_PDL)'
            pre='    pa_lines=tl.arange(0,BK//64)\n    _l2_prefetch(W+n[:,None]*K+kb0*BK+pa_lines[None,:]*64)\n'
            code=once(code,wait,pre+wait)
        changes.append((n.lineno-1,n.end_lineno,code+'\n'))
    for lo,hi,code in sorted(changes,reverse=True):lines[lo:hi]=[code]
    out=''.join(lines);ast.parse(out);return out


if __name__=='__main__':
    out=patch_hint((REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text())
    p=HERE/'hint_trial.py';p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest())
