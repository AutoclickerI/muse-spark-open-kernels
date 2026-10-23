from pathlib import Path
root=Path('problems/muse_prefill_layer')
source=(root/'cand_canonical_autotune.py').read_text()
a=source.index('def _attention(data, q8, st):')
b=source.index('\ndef kernel_fn(',a)
original=source[a:b].replace('def _attention(','def _attention_fa3(',1)
attention=(root/'dev/attention_fp8.py').read_text().split('ATTENTION_FP8_MIN_L =')[0]
source=source[:a]+original+'\n'+attention+'\n'+source[b:]
source += '\n'+(root/'dev/attention_fp8_autotune.py').read_text()
doc_end=source.index('"""',source.index('"""')+3)+3
source=source[:source.index('"""')]+'"""Prefill layer with one canonical V6 expert bank and prepare-time synthetic\nselection of both expert readers and long-context attention paths on Hopper."""'+source[doc_end:]
(root/'cand_prefill_autotune.py').write_text(source)
# Dev only: run both legal synthetic tuners on the native sm120 implementations.
(root/'dev/cand_prefill_autotune_forced_local.py').write_text(source.replace('if IS_HOPPER:\n        _tune_', 'if True:\n        _tune_'))
