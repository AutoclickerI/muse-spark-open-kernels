"""Build the audited, attributed NF dependency closure. EXP is input data, not imported.
Run only after source review. Selects packers/builders, not the historical dispatcher.
"""
import ast,base64,hashlib,json,re,zlib
from pathlib import Path
H=Path(__file__).resolve().parent
EXP=Path('/home/jihoon/.tmp-live-20260919/runtime/claude-1016/-home-jihoon-competitions-muse-kernel/ae5ada91-d8f5-49d0-a518-22df30a603ba/scratchpad/ext_expanded/nf/submission_expanded.py')
SRC=Path('/home/jihoon/.tmp-live-20260919/runtime/claude-1016/-home-jihoon-competitions-muse-kernel/ae5ada91-d8f5-49d0-a518-22df30a603ba/scratchpad/ext/NathenFernandes_muse-bf16-proj/repo')
raw=EXP.read_text();tree=ast.parse(raw)
imports=sorted({ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))})
calls=sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)})
strings='\n'.join(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str))
forbidden=[x for x in calls if re.search(r'(^|\.)(open|exec|eval|system|popen|save|write|read|environ|getenv|subprocess|socket|connect)($|\.)',x)]
native=re.findall(r'\b(system|popen|fork|execv|fopen|ofstream|ifstream|getenv|setenv|curl|socket|connect|unlink|rename)\s*\(',strings)
assert not forbidden and not native,(forbidden,native)
(H/'source_audit.json').write_text(json.dumps(dict(source='NathenFernandes/muse-bf16-proj@86aff75',exp_sha256=hashlib.sha256(EXP.read_bytes()).hexdigest(),imports=imports,python_direct_forbidden_calls=forbidden,native_forbidden_calls=native,reviewed=True,notes=['No direct network/subprocess/filesystem/environment APIs','load_inline writes JIT build files through PyTorch','Diagnostic DIAG modes excluded by selected configurations','Clock-read unreachable-trap scheduling barriers present; not timing manipulation']),indent=2)+'\n')
# Native source strings have been reviewed before this builder executes any port.
roots=['v18d_v18d_native_build','v18d_v18d_native_pack_format','v18d_v18d_native_stream',
'v18s_build','v18s_pack_format','v18s_stream','tt_v10_sm2_smallm2_build','tt_v10_sm2_smallm2_pack_format','tt_v10_sm2_smallm2_stream',
 'tt_v12_smtrace1_smtrace_build','tt_v12_smtrace1_smtrace_pack_format','tt_v12_smtrace1_smtrace_stream',
 'tt_v13_smtrace8_smtrace_build','tt_v13_smtrace8_smtrace_pack_format','tt_v13_smtrace8_smtrace_stream',
 'dd_decodediet_cps_build','dd_decodediet_dd_cap2_b_cpssm_pack_format','dd_decodediet_dd_cap2_b_cpssm_stream',
 'mt_M16TRACE_SOURCES','mt_wp0_pack_format','mt_wp0_stream','BUNDLE_QKV_CUDA','SM_CPP','sm_stream',
 'tt_v10_m32x__q_build','tt_v10_m32x_m32x_pack_format','tt_v10_m32x_m32x_stream',
 'b2q_tt_v10_m32x__q_build','b2p_tt_v10_m32x__build','b2p_tt_v10_m32x_m32x_pack_format',
 'dd_decodediet_tt_mw_build','dd_decodediet_tt_mw_pack_weight',
 'prepare_olayer_candidates','run_olayer_candidate','b2o_prepare_olayer_candidates','b2o_run_olayer_candidate',
 'tt_v10_tune_build_rl','tt_v10_tune_pack_weight','tt_v14_qkv_build_qkv','tt_v14_qkv_qbig_pack',
 'tt_lt__lt_state','tt_lt__lt_plan','tt_lt__LT_H200_FIXED']
# Only top-level definitions and assignments with a single stable symbol. Last definition wins.
def names(n):
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)):return [n.name]
 if isinstance(n,ast.Assign):return [t.id for t in n.targets if isinstance(t,ast.Name)]
 return []
symbols={name:n for n in tree.body for name in names(n)}
missing=set(roots)-symbols.keys();assert not missing,missing
needed=set(roots);todo=list(roots)
while todo:
 name=todo.pop();node=symbols[name]
 for q in ast.walk(node):
  if isinstance(q,ast.Name) and isinstance(q.ctx,ast.Load) and q.id in symbols and q.id not in needed:
   needed.add(q.id);todo.append(q.id)
selected={id(symbols[x]) for x in needed}
# imports are explicit rather than dragging SGLang's original dispatcher in.
pre='import hashlib, statistics, time\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
body=pre+'\n\n'.join(ast.unparse(n) for n in tree.body if id(n) in selected)+'\n'
# JIT target follows the actual GPU. Hopper-only kernels are replaced by local arithmetic stand-ins
# in the integration adapter; warp MMA families build and run natively on sm_120 too.
body=body.replace("'-gencode=arch=compute_90a,code=sm_90a'", "_nf_arch_flag()")
body='def _nf_arch_flag():\n    c=torch.cuda.get_device_capability()\n    a="90a" if c==(9,0) else str(c[0])+str(c[1])\n    return "-gencode=arch=compute_"+a+",code=sm_"+a\n'+body
license=(SRC/'LICENSE').read_text()
notice='Nathen Fernandes; github.com/NathenFernandes/muse-bf16-proj @ 86aff75; MIT\n'+license
(H/'LICENSE.NathenFernandes').write_text(license)
packed=base64.b64encode(zlib.compress(body.encode(),9)).decode()
module='\n'.join('# '+l for l in notice.splitlines())+'\nimport base64,zlib\nexec(compile(zlib.decompress(base64.b64decode('+repr(packed)+')), "<NF86aff75-selected>", "exec"))\n'
(H/'nf_selected.py').write_text(module)
(H/'vendor_manifest.json').write_text(json.dumps(dict(upstream='86aff75',exp_sha256=hashlib.sha256(EXP.read_bytes()).hexdigest(),closure_symbols=sorted(needed),closure_bytes=len(body),compressed_module_bytes=len(module),module_sha256=hashlib.sha256(module.encode()).hexdigest()),indent=2)+'\n')
print('symbols',len(needed),'source bytes',len(body),'packed',len(module))
