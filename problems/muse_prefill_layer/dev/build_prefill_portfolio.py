from pathlib import Path
P=Path(__file__).resolve().parents[1]
s=(P/'cand_expert_portfolio.py').read_text()
s=s.replace('def _attention(data, q8, st):','def _attention_fa3(data, q8, st):',1)
attention=(P/'dev/attention_fp8.py').read_text().split('ATTENTION_FP8_MIN_L =')[0]
a=s.index('\ndef kernel_fn(')
s=s[:a]+'\n'+attention+'\n'+s[a:]
old=(P/'cand_prefill_autotune.py').read_text()
tuning=old[old.index('# Four complete attention paths'):]
s+= '\n'+tuning
s=s.replace('Prefill expert portfolio: canonical NVFP4 bank, BN64/128/256 and timed BF16 scratch.',
    'Prefill portfolio: one canonical bank, BN64/128/256/DG expert selection and FP8 attention.')
(P/'cand_prefill_portfolio.py').write_text(s)
# Fixed v11-style family control isolates the serving bank/dequant change.
h=(P/'cand_expert_portfolio.py').read_text()
h=h.replace("{753:'adaptive128',3012:'bn128_r2',16384:'dg_hint'}", "{753:'pipeline',3012:'pipeline',16384:'dg_hint'}")
h=h.replace("    if IS_HOPPER:\n        _tune_portfolio(st)\n",'')
(P/'cand_canonical_hybrid.py').write_text(h)
print('built full portfolio and fixed canonical hybrid')
