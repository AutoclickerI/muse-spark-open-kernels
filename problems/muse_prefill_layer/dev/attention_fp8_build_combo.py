from pathlib import Path
root=Path('problems/muse_prefill_layer')
for base,name in (('cand_canonical_prefill.py','cand_canonical_attention.py'),
                  ('cand_canonical_grouped.py','cand_grouped_attention.py')):
    source=(root/base).read_text()
    a=source.index('def _attention(data, q8, st):')
    b=source.index('\ndef kernel_fn(',a)
    original=source[a:b].replace('def _attention(','def _attention_fa3(',1)
    addition=(root/'dev/attention_fp8.py').read_text()
    addition=addition.replace('return _attention_fp8(data, q8, st)','return _attention_fp8(data, q8, st, contig=True)')
    combined=source[:a]+original+'\n'+addition+'\n'+source[b:]
    doc_end=combined.index('"""',combined.index('"""')+3)+3
    combined=combined[:combined.index('"""')]+'"""Prefill layer with one canonical V6 NVFP4 expert bank and lossless-gather\nFP8 QK/PV attention on long full layers; measured FA3 remains on other Hopper shapes."""'+combined[doc_end:]
    (root/name).write_text(combined)
