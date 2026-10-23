"""Assemble reviewed components into a fresh candidate; never edits submitted sources."""
import argparse,re
from pathlib import Path
p=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser()
a.add_argument('output')
a.add_argument('--base',required=True,choices=['v2','v3','v6','v7','v8'])
a.add_argument('--attention',choices=['paged','contiguous','unpacked'],default='paged')
a.add_argument('--dense',choices=['cublas','deepgemm'],default='cublas')
a.add_argument('--compact-state',action='store_true')
args=a.parse_args()
out=p/args.output
if out.exists():raise SystemExit('Refusing to overwrite '+str(out))
s=(p/f'cand_{args.base}.py').read_text()
# Remove inherited untimed diagnostic; it is disabled in these candidates already.
i=s.index('def _bench_w4a16('); j=s.index('\ndef prepare',i)
s=s[:i]+s[j+1:]
s=s.replace('    if False:\n        _bench_w4a16(weights)\n','')
if args.attention!='paged':
 s=s.replace('attn = _attention(data, q8, st)','attn = _attention_contiguous(data, q8, st)')
 att=(p/'dev/attention_dense.py').read_text()
 if args.attention=='unpacked':
  att=att.replace("window_size=(W-1,0) if W else (-1,-1), **descale)", "window_size=(W-1,0) if W else (-1,-1),\n                pack_gqa=False if B*triton.cdiv(L,128)*HQ >= 132 else None,\n                num_splits=1 if B*triton.cdiv(L,128)*HQ >= 132 else 0, **descale)")
 s+='\n\n'+att
if args.dense=='deepgemm':
 for x,w in [('x','qkv'),('o_in','o'),('h','pooler'),('moe','unpooler')]:
  s=s.replace(f'torch.mm({x}, st["{w}"].t())',f'_dense_mm({x}, st["{w}"])')
 d=(p/'cand_v5.py').read_text();s+='\n\n'+d[d.index('def _dense_mm('):]
if args.compact_state and args.base in ('v3','v6','v7'):
 # These expert paths cover every T; they need only their own packed bank and scales.
 i=s.index('    try:\n        st["bank"] = _marlin_bank(weights)')
 j=s.index('    torch.cuda.synchronize()',i)
 s=s[:i]+'    st["bank"] = None\n'+s[j:]
 # Remove raw packed weights only after their lossless replacement has been built.
 start=s.rindex('def prepare(weights):')
 end=s.index('    return st',start)
 s=s[:end]+'''    for key in ('w13', 'w13_sf', 'w2', 'w2_sf'):
        st.pop(key, None)
'''+s[end:]
s=s.replace('Muse Spark Open prefill layer: v1.',f'Muse Spark Open prefill layer: {out.stem}; {args.base} experts, {args.attention} attention, {args.dense} dense.')
out.write_text(s)
print(out)
