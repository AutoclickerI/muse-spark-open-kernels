"""Pin the completed local evidence to the current candidate."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
def read(path):return json.loads(path.read_text())
sha=hashlib.sha256((ROOT/'problems/muse_decode_layer/cand_r3_best6.py').read_bytes()).hexdigest()
manifest=read(HERE/'build_manifest.json');audit=read(HERE/'audit.json')
assert sha==manifest['sha256']==audit['sha256']
screens={}
for kind in ('normal','forced'):
    data=read(HERE.parent/f'best6_attention_{kind}_local.json')
    assert data['sha256']==sha
    for key,count in (('tests',13),('shapes',12),('static',14)):
        assert len(data[key])==count and all(r['ok'] for r in data[key]),(kind,key)
    screens[kind]=dict(sha256=sha,prepare_s=data['prepare_s'],tests=13,graphs=12,static=14,
                       gpu=data['gpu'],cc=data['cc'])
forced=read(HERE/'forced_choices_local.json')
assert len(forced['rows'])==12
for row in forced['rows']:
    assert len(row['seeds'])==3 and len(row['seq_lens'])==3
    assert all(len(t['seed_us'])==3 for t in row['trials'])
    assert any(t['choice']==['layer',None] for t in row['trials'])
assert all(v>0 for row in forced['base']['cycle']['choices'].values() for v in row.values())
log=(HERE.parent/'best6_attention_forced_cold_local.log').read_text()
cold=float(re.search(r'BEST_PREPARE elapsed=([\d.]+)s',log)[1])
assert cold<240
regression=read(HERE/'regression_local.json');equivalence=read(HERE/'public_equivalence_local.json')
assert equivalence['fixtures']==36 and equivalence['bitwise_equal']
assert regression['mean_over_draws_rejects_single_sample_winner']
result=dict(sha256=sha,manifest=manifest,screens=screens,cold_candidate_prepare_s=cold,
            exact_unchanged_base_functions=len(audit['unchanged_functions']),
            all_native_sources_unchanged=True,public_generator_functions=audit['exact_public_generator_functions'],
            attention_shapes=12,draws_per_shape=3,public_fixture_equivalence=equivalence,
            regression=regression,forced_attention_rows=forced['rows'],forced_experts=forced['base'],
            prior_expert_error_audit='best6/validation.json: original c582, 18 injected RuntimeErrors; expert helpers unchanged',
            h200=dict(best5_job='691bff0bcff44e99',best5_score=1.353449,
                      revised_candidate_sent=False,native_adapter_safety='pending moe_decode_adapters on MoE host',
                      owner_hold='no best6 send until M128/256 native adapter lab passes'))
(HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(sha256=sha,screens=screens,cold_candidate_prepare_s=cold,
                      public_fixtures=equivalence['fixtures'],attention_shapes=12),indent=2))
