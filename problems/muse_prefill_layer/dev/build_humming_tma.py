"""Copy the sibling's measured TMA indexed configuration into a complete prefill candidate."""
from pathlib import Path
p=Path(__file__).resolve().parents[1]
s=(p/'cand_v6.py').read_text()
s=s.replace('Muse Spark Open prefill layer: v1.', 'Muse Spark Open prefill layer: v14, TMA indexed NVFP4 experts.')
start=s.index('    try:\n        st["bank"] = _marlin_bank(weights)')
end=s.index('    torch.cuda.synchronize()',start)
s=s[:start]+s[end:]
s=s.replace('    if False:\n        _bench_w4a16(weights)\n','')
s=s.replace('''        return json.dumps(config)
''','''        if torch.cuda.get_device_capability()[0] == 9:
            # Exact configuration checked by the MoE owner in cand_humming3.py.
            # Indexed activation gather/scatter stays direct; weights/scales use TMA.
            config.update(use_tma=True, use_tma_a=False, use_tma_as=False, use_tma_c=False,
                          use_mbarrier=True, use_warp_spec=True)
        return json.dumps(config)
''')
s=s.replace("    st['humming'] = _Experts(weights, 0, 32)\n", "    st['humming'] = _Experts(weights, 0, 32)\n    for key in ('w13', 'w13_sf', 'w2', 'w2_sf'):\n        st.pop(key, None)\n")
# Remove dead expert backends and their one-off prepare diagnostic.
for first,last in [('def _marlin_bank(', 'def prepare_base('), ('def _experts_marlin(', 'def kernel_fn(')]:
 i=s.index(first);j=s.index(last,i);s=s[:i]+s[j:]
out=p/'cand_v14.py'
if out.exists(): raise SystemExit('Refusing to overwrite '+str(out))
out.write_text(s)
print(out)
