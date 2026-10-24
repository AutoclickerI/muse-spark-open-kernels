"""Pin the screened candidate and all exercised table choices."""
from pathlib import Path
import argparse,ast,hashlib,json
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
read=lambda p:json.loads(p.read_text())
parser=argparse.ArgumentParser();parser.add_argument('--reset',action='store_true');args=parser.parse_args()
prefix='reset_' if args.reset else ''
manifest=read(HERE/(prefix+'manifest.json'));source=ROOT/manifest['source']
sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==manifest['sha256']
audit=read(HERE/(prefix+'audit_local.json'));assert audit['sha256']==sha
screens={}
for mode in ('normal','forced_default','forced'):
    screen=read(HERE/(prefix+mode+'_local.json'));assert screen['sha256']==sha
    for key,count in (('tests',13),('shapes',12),('static',18)):
        assert len(screen[key])==count and all(row['ok'] for row in screen[key]),(mode,key)
    screens[mode]=dict(sha256=sha,tests=13,graphs=12,static=18,prepare_s=screen['prepare_s'],
                       gpu=screen['gpu'],cc=screen['cc'])
forced=read(HERE/(prefix+'forced_dispatch_local.json'))['forced']
assert forced['stream_ops']==24
assert sum(key.startswith("('projection',") for key in forced['gates'])==24
assert {p['glue'] for key,p in forced['plans'].items() if key!='other'}=={'best5','best6'}
assert all(count>0 for row in forced['full_layer_experts'].values() for count in row.values())
focus=ROOT/'problems/muse_decode_layer/FOCUS.md';assert len(focus.read_text().splitlines())<=30
table=next(n.value for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='BEST7_SETTINGS' for t in n.targets))
table=ast.literal_eval(table)
assert len(table)==13 and table[32,8,0][2]==table[32,8,2048][2]=='humming_masked'
assert all(row[2]=='v9nt1' for key,row in table.items() if key!='other' and key[0]*key[1]<256)
assert table[1,1,0][1]==table[1,1,2048][1]==('attn_main',None)
assert all(row[3]=='auto' for row in table.values()),'pending streaming must remain off in the default'
default=read(HERE/(prefix+'forced_default_dispatch_local.json'))['forced']
assert default['stream_ops']==0 and default['native_calls']['humming']>0
assert default['humming_raw_scales']==dict(w13=True,w2=True)
precision=read(HERE/(prefix+'precision_gate_local.json'));assert precision['sha256']==sha
assert precision['finite_scale_codes']==254 and all(precision[key] for key in (
    'canonical_raw_bank_exact','timed_expansion_exact','raw17_scale_bytes_preserved','raw17_nvfp4_codes_recovered_exactly'))
for key,plan in default['plans'].items():
    if key=='other':continue
    assert plan['moe']==('humming_masked' if key.startswith('(32, 8,') else 'v9nt1'),(key,plan)
result=dict(manifest=manifest,screens=screens,source_audit=audit,default_settings={str(k):v for k,v in table.items()},
            forced_settings=forced,forced_default_settings=default,raw_scale_admission=precision,
            native_h200_composition='unverified; optional library launches use portable stand-ins locally',
            native_humming_evidence=dict(lab='de6a56fc38e9477d',full_layer_check='1b80ffc037cc454c',
                mean_12seeds_m256=dict(humming=448.04,nt2=545.46),prepare_m256=dict(humming=419.971,nt2=511.784)),
            current_score_model=read(HERE/'sensitivity.json'),focus_lines=len(focus.read_text().splitlines()),
            receipt_verdicts=dict(paired=dict(job='1365af6696884c7e',verdict='failed route IDs; owner re-screening correction'),
                small_m=dict(job='b68afb72b72542b4',verdict='retain NT1'),
                streaming=dict(job='feb7b287e3464e4a',verdict='disabled pending in-layer overlap comparison')),
            sent=False)
(HERE/(prefix+'validation.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(sha256=sha,screens=screens,forced_stream_ops=24,forced_gates=len(forced['gates']),focus_lines=result['focus_lines']),indent=2))
