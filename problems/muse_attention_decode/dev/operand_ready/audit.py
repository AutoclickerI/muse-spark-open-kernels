"""Check the proposed shared-full K/V split against the actual S15 dispatch."""
import ast, hashlib, json, re
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]
SOURCE = R/'problems/muse_decode_layer/cand_r3_best7_s15.py'
SHA = '602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'

def main():
    text = SOURCE.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == SHA
    tree = ast.parse(text)
    assignments = {getattr(t, 'id', ''): n for n in tree.body if isinstance(n, ast.Assign) for t in n.targets}
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    value = lambda name: ast.literal_eval(assignments[name].value)
    code = lambda name: ast.get_source_segment(text, functions[name])
    assert value('_VQ8_ROWS') == {(16, 8, False): (4, 8), (32, 8, False): (4, 8)}
    assert value('_CLUSTER_ROWS') == ((1, 1, True),)
    assert all(not row[2] for row in value('_PP_ROWS'))
    assert "if vq8 is not None:return _vq8_attention" in code('_best7_attention')
    assert "return _attention_fa3_public(q,gate,data,state)" in code('_attn_port_variant')
    assert 'flash_attn_with_kvcache' in code('_attention_fa3_public')
    # No alternate assignment can silently enable VQ8 on a sliding row.
    writes = [ast.get_source_segment(text, n) for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript)
              and isinstance(t.slice, ast.Constant) and t.slice.value == 'vq8_rows' for t in n.targets)]
    assert writes == ["state['vq8_rows']=_vq8_prepare(device)"]
    assert 'return _VQ8_ROWS' in code('_vq8_prepare')
    kernels = {}
    for name in ('cs_CUDA', 'CUDA_SRC', 'VQ8_CUDA'):
        source = value(name)
        hits = [dict(line=i, code=line.strip()) for i, line in enumerate(source.splitlines(), 1)
                if re.search(r'mb_(wait|arrive)\(&full\[', line)]
        kernels[name] = dict(sha256=hashlib.sha256(source.encode()).hexdigest(),
                             source_assignment_line=assignments[name].lineno, full_barrier_uses=hits)
    assert not kernels['cs_CUDA']['full_barrier_uses']
    assert not kernels['CUDA_SRC']['full_barrier_uses']
    vq8 = value('VQ8_CUDA')
    assert 'transpose_group(VR + slot * 16384, VT + slot * 16384, pw, lane);' in vq8
    assert 'if (lane == 0) mb_arrive(&full[slot]);' in vq8
    assert 'mb_wait(&full[(rr + 1) % ST], ((rr + 1) / ST) & 1);' in vq8
    for b in (1, 16, 32):
        assert value('BEST7_SETTINGS')[(b, 8, 2048)][1] == ('layer', None)
    decisions = [
        dict(item='decode1 sliding, last-CTA', verdict='drop shared-full split claim',
             evidence='cs_CUDA sliding_main uses cp_wait<1>, __syncthreads, trans64 then QK in the same warpgroup; no full ring',
             leaves_open='Reordering the serial transpose/QK schedule is a different scheduling experiment'),
        dict(item='decode16/32 sliding, attn_main', verdict='drop shared-full split claim',
             evidence='CUDA_SRC loop uses cp_wait<STAGES-2>, __syncthreads then compute_tile; no full ring',
             leaves_open='A new producer/consumer pipeline, rather than splitting an existing barrier'),
        dict(item='verify1/16/32 sliding, including claimed verify32 VQ8', verdict='drop VQ8 sliding claim',
             evidence='_VQ8_ROWS contains only (16,8,False)/(32,8,False); all sliding verify rows select layer -> FA3',
             leaves_open='FA3 preparation/main-kernel restructuring; this audit does not establish its internal barriers'),
        dict(item='VQ8 full verify16/32', verdict='confirmed, owned by P5b full-attention lane',
             evidence='producer transposes V before arriving full[slot]; consumer waits full[(rr+1)%ST] before next QK',
             leaves_open='P5b separate K/V-ready arms; not a sliding dispatch'),
    ]
    result = dict(s15=str(SOURCE.relative_to(R)), sha256=SHA,
                  source_evidence={name: functions[name].lineno for name in
                     ('_best7_attention', '_attn_port_variant', '_attention_fa3_public', '_vq8_prepare')},
                  vq8_rows=[list(k)+list(v) for k,v in value('_VQ8_ROWS').items()],
                  kernels=kernels, decisions=decisions,
                  sliding_arms_built=[], sliding_arms_screened=[], sliding_payloads_sent=[],
                  reason='No reviewed shared-full split exists on any active S15 sliding path; no no-effect arm is built')
    timeline = R/'problems/muse_decode_layer/dev/s15_timeline/out/h200/timeline.json'
    if timeline.exists():
        native = json.loads(timeline.read_text())
        assert native['s15_sha256'] == SHA
        result['native_dispatch'] = dict(receipt='7eaa262a30ec483e', kernels={c['case']:
            [n['name'] for n in c['trace'] if n['role']=='attention_sliding']
            for c in native['cases'] if c['case'].endswith('S')})
    (H/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    for item in decisions:
        print(item['item']+': '+item['verdict'])

if __name__ == '__main__':
    main()
