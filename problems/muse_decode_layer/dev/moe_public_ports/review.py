"""Source-only provenance audit. External files are read, never imported."""
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;D=Path('/data2/jihoon/muse_kernel/review/moe-public-Ury55uMA')
s=(D/'operatorstack/muse_moe_w4a16/submission.py').read_text();p=(H/'operator.py.inc').read_text()
def kernels(s):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.decorator_list}
a,b=kernels(s),kernels(p);assert a==b
lee=(D/'lee101/muse_moe_w4a16/submission_v43d.py').read_text();t=ast.parse(lee)
val=lambda name:ast.literal_eval(next(n for n in t.body if isinstance(n,ast.Assign) and any(getattr(x,'id','')==name for x in n.targets)).value)
f=(H/'lee_f.cu.inc').read_text();orig=val('CUDA_F');core=orig[orig.index('#define HD'):orig.index('static int g_gridF')];assert core in f
x=(H/'lee_x.cu.inc').read_text();origx=val('CUDA_X');a=origx.index('template <int N, bool FG>');b=origx.index('__global__ void __launch_bounds__(NTH, 1) moe_x');assert origx[a:b] in x
forbidden=('subprocess','os.system','socket','requests','urllib','eval(')
for key in ('operator.py.inc','lee_repack.py.inc','lee_x.cu.inc','lee_f.cu.inc'):
 text=(H/key).read_text();assert not any(v in text for v in forbidden),(key,'unsafe driver')
report=dict(passed=True,isolated_clones=str(D),operator_identical_device_AST=sorted(a for a in kernels(p)),lee_F_identical_device_core_sha256=hashlib.sha256(core.encode()).hexdigest(),lee_X_identical_math='group/unit/decode and all decode/WGMMA primitives retained; phase queue range/dependency/finalizer are explicit specializations',changed_host='owned raw pointer API, CUDA launch error checks, unique C++ namespace per configuration',unexecuted='donor scripts/tests/notebooks/autotuners/diagnostics',licenses='Full MIT, operatorstack NOTICE, Apache-2.0 text embedded in copied Python and payload; Lee pin has no NOTICE')
(H/'review.json').write_text(json.dumps(report,indent=2)+'\n');print('REVIEW PASS',len(kernels(p)),'identical operator device functions')
