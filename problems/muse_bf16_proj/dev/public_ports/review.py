"""Copy only reviewed kernel definitions from an unexecuted, pinned external clone."""
import ast,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];CLONE=Path('/data2/jihoon/tmp/p10_operator.UVMtuP');PIN='6d9b77656b6c295e421be9e1424b0f8f99299b0a'
def show(path):return subprocess.check_output(['git','-C',str(CLONE),'show',PIN+':'+path],text=True)
s=show('muse_bf16_proj/submission.py');assert hashlib.sha256(s.encode()).hexdigest()=='0808a0f1fe5404808382f5160f482e636d1c57eb68b9e66db3cc7d35c0d99229'
keep={'_cbf_skinny_gemm','_cbf_skinny_gemm_tma','_cbf_skinny_gemm_tma_tiled','_cbf_gemv_rows','_cbf_gemv32','_cbf_cluster_gemm','_cbf_v4copy','_cbf_lm_partials','_cbf_lm_partials_tiled'};constants={'CBF_V4COPY2','CBF_V4COPY4'};nodes=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in keep or isinstance(n,ast.Assign) and any(getattr(t,'id','') in constants for t in n.targets)]
text='import torch,triton\nimport triton.language as tl\nfrom triton.tools.tensor_descriptor import TensorDescriptor\n\n'+'\n\n'.join(s.splitlines(keepends=True)[n.lineno-2].strip()+'\n'+ast.get_source_segment(s,n) if isinstance(n,ast.FunctionDef) and n.decorator_list else ast.get_source_segment(s,n) for n in nodes)+'\n'
# Functions are pure kernels/ASM generators; no donor launch/prepare/fallback/driver is retained.
assert not any(isinstance(n,(ast.ImportFrom,ast.Import)) and ('sglang' in ast.get_source_segment(text,n)) for n in ast.parse(text).body)
license=show('LICENSE');nf=(R/'problems/muse_bf16_proj/dev/public_nf/LICENSE.NathenFernandes').read_text();notice='Operatorstack / bigboateng, '+PIN+'\n'+license+'\nNathenFernandes/muse-bf16-proj @86aff75 (existing reviewed S19 vendor)\n'+nf
(H/'NOTICES.txt').write_text(notice);(H/'operatorstack.py.inc').write_text('\n'.join('# '+line for line in ('Operatorstack / bigboateng MIT\n'+license).splitlines())+'\n'+text)
meta=dict(repo='https://github.com/operatorstack/open-kernel-challenge-solutions',pin=PIN,external_clone=str(CLONE),checkout_executed=False,donor_scripts_tests_executed=False,source_sha256=hashlib.sha256(s.encode()).hexdigest(),license_sha256=hashlib.sha256(license.encode()).hexdigest(),reviewed_entire_source_lines=len(s.splitlines()),retained=[dict(name=getattr(n,'name','constants'),start=n.lineno,end=n.end_lineno) for n in nodes],omitted='All donor import fallbacks, dispatch, prepare, fixed descriptors, driver/tests. Reviewed pure kernels only.',NF='Use the already-reviewed MIT86aff75 vendor behind S19, including unshipped default v18 site dispatches; no donor test/build script executed')
(H/'review.json').write_text(json.dumps(meta,indent=2)+'\n');print('reviewed',len(s.splitlines()),'lines; retained',len(nodes),'pure definitions')
