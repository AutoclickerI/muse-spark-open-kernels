"""Receipt arithmetic only; no serving latency measurement or interpolation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
receipt=next((ROOT/'runs/muse_bf16_proj').glob('*f3786341611f4f1f.json'))
data=json.loads(receipt.read_text())['result']
assert data['passed'] and not data['flags']
records={(r['case']['layer'],r['case']['m']):r for r in data['records'] if 'speedup' in r}
rows=[]
for c,m in ((1,8),(4,32),(16,128),(32,256)):
    measured={name:{'sglang_us':records[(name,m)]['baseline_ns']/1000,
                    'receipt_us':records[(name,m)]['median_ns']/1000}
              for name in ('qkv_gate','o','pooler','unpooler','router')}
    def delta(name):return measured[name]['sglang_us']-measured[name]['receipt_us']
    # Give unchanged cuBLAS paths zero credit rather than rewarding run noise.
    active=('pooler','unpooler') if m in (8,256) else ('o','pooler','unpooler') if m==128 else ()
    dense_ms=.062*sum(delta(name) for name in active)
    existing_ms=.062*(delta('router')+(delta('qkv_gate') if m==8 else 0))
    rows.append(dict(concurrency=c,m=m,per_shape_us=measured,
                     dense_three_sglang_us=sum(measured[n]['sglang_us'] for n in ('o','pooler','unpooler')),
                     dense_three_receipt_us=sum(measured[n]['receipt_us'] for n in ('o','pooler','unpooler')),
                     dense_three_credited_ms=dense_ms,router_qkv_credited_ms=existing_ms,
                     combined_credited_ms=dense_ms+existing_ms,
                     all_five_sglang_us=sum(v['sglang_us'] for v in measured.values()),
                     all_five_receipt_us=sum(v['receipt_us'] for v in measured.values())))
report=dict(receipt=str(receipt.relative_to(ROOT)),kind='standalone receipt model, not serving measurement',
            layers=62,rows=rows,unmeasured_shapes=['pooler N512','unpooler N1024 above M8'])
output=ROOT/'runs/screen/serving_proj/receipt_model.json'
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
