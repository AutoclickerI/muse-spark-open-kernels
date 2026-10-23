"""Use expected live rows to tune grouped GEMM; preserve worst-case dynamic capacity."""
from pathlib import Path
p=Path(__file__).resolve().parents[1]
s=(p/'cand_v11.py').read_text()
s=s.replace('Muse Spark Open prefill layer: v1.', 'Muse Spark Open prefill layer: v16, shape-derived grouped GEMM row hint.')
s=s.replace('use_psum_layout=True)', 'use_psum_layout=True, expected_m_for_psum_layout=max(1, triton.cdiv(T, LOCAL_EXPERTS)))')
# Retain measured v3 small-T and v2 large-T dispatch. Tune only its grouped GEMMs.
out=p/'cand_v16.py'
if out.exists():raise SystemExit('Refusing to overwrite '+str(out))
out.write_text(s)
# Force this branch through every official local test in addition to the actual candidate.
(p/'dev/cand_v16_dg_local.py').write_text(s.replace('if T < 8192:', 'if T < 0:'))
print(out)
