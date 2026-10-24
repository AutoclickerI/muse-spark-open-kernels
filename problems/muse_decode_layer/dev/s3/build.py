"""Replace S2's M256 producer with the measured lossless C1g2112 producer."""
from pathlib import Path
import ast
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
base = R / 'problems/muse_decode_layer/cand_r3_best7_scale_w2fix.py'
payload = R / 'prof/lab/moe_recoded_m256.py'
s = base.read_text()
p = payload.read_text()
assert hashlib.sha256(s.encode()).hexdigest() == 'b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337'
assert hashlib.sha256(p.encode()).hexdigest() == 'd30f4a862fa0b882529ba3b9ea9c37013f305559c707e27db62e10b4ff1a2bce'

def segment(text, node):
    lines = text.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return ''.join(lines[first - 1:node.end_lineno])

# Remove the replaced engine, including its raw-bank packing and admission gate.
removed = {'fx_module', '_s2_gate_moe', 'mw_repack_planes', 'fx_prepare',
           'fx_from_packed', 'fx_plan', 'fx_work', 'fx_call'}
lines = s.splitlines(keepends=True)
for node in reversed(ast.parse(s).body):
    if isinstance(node, ast.FunctionDef) and node.name in removed or (
        isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in
        ('FX_CUDA', 'FX_CPP', 'FX_CONFIGS') for t in node.targets)
    ):
        first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        del lines[first - 1:node.end_lineno]
s = ''.join(lines)
for old, new, count in (
    ('w2fix_cta1g2112', 'recoded_c1g2112', 6),
    ('fx_call(data,state[\'s2_bank\'],1,2112,1)', 'rc_call(data,state[\'s3_bank\'],1,2112)', 1),
    ('_s2_gate_moe(tokens,length,state,weights)', '_s3_gate_moe(tokens,length,state,weights)', 1),
    ('s2_future=native_pool.submit(fx_module)', 's3_future=native_pool.submit(rc_module)', 1),
    ("state['s2_mod']=s2_future.result()", "state['s3_mod']=s3_future.result()", 1),
    ('s2_started=time.monotonic()', 's3_started=time.monotonic()', 1),
    ("state['s2_bank']=fx_prepare(weights)", "state['s3_bank']=rc_prepare(weights)[1]", 1),
    ("phases['s2_repack']=time.monotonic()-s2_started", "phases['s3_repack']=time.monotonic()-s3_started", 1),
):
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
s = s.replace('# H200 97be11d4631b44d6: exact raw-bank CTA1g2112 producer and finalizer.',
              '# H200 9e8a6a15f95748db: exact lossless recoded C1g2112 producer and finalizer.')
names = {'mw_repack_planes', 'rc_masks', 'rc_repack', 'rc_prepare', 'rc_plan', 'rc_call'}
tree = ast.parse(p)
constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and len(n.targets) == 1 and
             getattr(n.targets[0], 'id', '') in ('RC_CUDA', 'RC_CPP', 'RC_CONFIGS')}
s += '\nimport hashlib\n' + '\n'.join(k + '=' + repr(v) for k, v in constants.items()) + '\n'
s += (H / 'runtime.py.inc').read_text() + '\n'
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        s += '\n' + segment(p, node) + '\n'
        if node.name == 'rc_masks':
            s += 'RC_SIGNS, RC_ZEROS = rc_masks()\n'

out = base.with_name('cand_r3_best7_scale_recoded_m256.py')
out.write_text(s)
sys.path.insert(0, str(R / 'tools'))
from okc_run import rejected_words
assert not rejected_words(s), rejected_words(s)
manifest = dict(source=str(out.relative_to(R)), sha256=hashlib.sha256(s.encode()).hexdigest(),
                base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                payload=str(payload.relative_to(R)), payload_sha256=hashlib.sha256(p.encode()).hexdigest(),
                rc_cuda_sha256=hashlib.sha256(constants['RC_CUDA'].encode()).hexdigest(),
                receipt='9e8a6a15f95748db', variant='C1g2112',
                config=dict(which=1, grid=2112, encoded=True), functions=sorted(names))
(H / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
