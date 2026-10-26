"""S19 own FA3 launcher binding, parallel row reduce, fixed-geometry VQ8 fusion."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'chain';o.mkdir(exist_ok=True)
b=json.loads((H/'phases/base_bundle.json').read_text())
header=(H/'chain_fa_params.hpp.inc').read_text()
bridge=(H/'chain_fa.cpp.inc').read_text().replace('// PARAM_HEADER',header).replace('run:combine_fn','combine_fn')
(o/'fa.cpp').write_text(bridge);(o/'reduce.cu').write_text((H/'chain_reduce.cu.inc').read_text())
(o/'fa_params.hpp').write_text(header)
meta=dict(base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),bridge_sha256=hashlib.sha256(bridge.encode()).hexdigest(),reduce_sha256=hashlib.sha256((o/'reduce.cu').read_bytes()).hexdigest(),FA3_source='https://github.com/sgl-project/sgl-flash-attn/tree/f89bc23066',abi_size=728,run_symbol='_Z11run_mha_fwdR16Flash_fwd_paramsP11CUstream_st')
(o/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
