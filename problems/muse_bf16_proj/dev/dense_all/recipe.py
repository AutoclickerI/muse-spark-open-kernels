"""The exact helper body shared by the matrix payload and production pruner."""
import ast,base64,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def helper_body():
    vendor=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text();call=next(n for n in ast.walk(ast.parse(vendor)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b64decode');source=zlib.decompress(base64.b64decode(ast.literal_eval(call.args[0]))).decode()
    functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    gpu=[]
    for name in ('v18s_c12_bytes','dd_decodediet_tt_mw_sm_c12_bytes'):
        f=functions[name];f=f.replace('hist = torch.bincount(eh.flatten(), minlength=128).cpu()','hist = torch.bincount(eh.flatten(), minlength=128)');f=f.replace('window = torch.stack([hist[b:b + 8].sum() for b in range(121)])','window = hist.unfold(0,8,1).sum(1)');assert '.cpu()' not in f;gpu.append(f)
    f=functions['dd_decodediet_tt_mw_pack_weight'];assert 'total * rw, 64)' in f and 'max_esc > 64' in f
    gpu.append(f.replace('total * rw, 64)','total * rw, 256)').replace('max_esc > 64','max_esc > 256'))
    body='\nimport time,hashlib,torch\nfrom torch.utils.cpp_extension import load_inline\n_nf_vendor=globals()["_nf_vendor"]\n_nf_group_pack=globals()["_nf_group_pack"]\nDA_GPU_PACK_SOURCE='+repr('\n\n'.join(gpu))+'\n'+(H/'runtime.py.inc').read_text()
    return body
