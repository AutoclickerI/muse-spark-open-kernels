from pathlib import Path
root = Path('problems/muse_prefill_layer')
source = (root / 'cand_v3.py').read_text()
a = source.index('def _attention(data, q8, st):')
b = source.index('\ndef _experts_marlin(', a)
original_attention = source[a:b].replace('def _attention(', 'def _attention_fa3(', 1)
source = source[:a] + original_attention + '\n' + (root / 'dev/attention_fp8.py').read_text() + '\n' + source[b:]
doc_end=source.index('"""',source.index('"""')+3)+3
source=source[:source.index('"""')]+'"""Prefill layer with paged FP8 QK/PV attention on long full layers, FA3 on other\nHopper shapes, BF16 dense projections, and the measured v3 BN128 WGMMA expert path."""'+source[doc_end:]
(root / 'cand_attention_fp8.py').write_text(source)

forced = source.replace('ATTENTION_FP8_MIN_L = 2048', 'ATTENTION_FP8_MIN_L = 0').replace("if cfg['window'] == 0 and cfg['q_len'] >= ATTENTION_FP8_MIN_L:", "if cfg['q_len'] >= ATTENTION_FP8_MIN_L:")
(root / 'dev/cand_attention_fp8_forced_local.py').write_text(forced)
contig = source.replace('return _attention_fp8(data, q8, st)', 'return _attention_fp8(data, q8, st, contig=True)')
(root / 'cand_attention_fp8_contig256.py').write_text(contig.replace('return _attention_fp8(data, q8, st, contig=True)', 'return _attention_fp8(data, q8, st, bn=256, contig=True)'))
(root / 'cand_attention_fp8_contig.py').write_text(contig)
(root / 'dev/cand_attention_fp8_contig_forced_local.py').write_text(forced.replace('return _attention_fp8(data, q8, st)', 'return _attention_fp8(data, q8, st, contig=True)'))
(root / 'dev/cand_attention_fp8_contig256_forced_local.py').write_text(forced.replace('return _attention_fp8(data, q8, st)', 'return _attention_fp8(data, q8, st, bn=256, contig=True)'))
