"""Build structural dense/router fusion and expert/unpooler overlap, preserving queued sources."""
from pathlib import Path
import ast,hashlib
from builder_safety import guarded_tuners
P=Path(__file__).resolve().parents[1]
source=P/'cand_fused_expert_portfolio.py';s=source.read_text();queued_hash=hashlib.sha256(source.read_bytes()).hexdigest()
s=guarded_tuners(s)
s=s.replace('    _prepare_overlap(st)\n','    _prepare_overlap(st)\n    _prepare_dense_fusion(st)\n    st[\'dense_router_workspace\']=torch.empty(256*1024*1024,device=st[\'qkv\'].device,dtype=torch.uint8)\n    st[\'_router_original\']=weights[\'router\'].float()\n',1)
s=s.replace("st['expert_dispatch']={753:'fused_bf16_64',3012:'fused_bf16_64',16384:'fused_bf16_64'}", "st['expert_dispatch']={753:'raw11',3012:'raw11',16384:'raw_v11'}")
s=s.replace('        _tune_score_experts(st,flush)\n','        _tune_dense_structures(st,flush)\n        _tune_score_experts(st,flush)\n        _tune_dense_structures(st,flush,back=True)\n',1)
s=s.replace("    st.pop('_generator');st.pop('_deadline')", "    st.pop('_router_original');st.pop('_generator');st.pop('_deadline')")
# Replace the attention output projection/boundary and the whole router/expert/back schedule.
a=s.index('    a = _score_dense(o_in, st, "o")',s.index('def kernel_fn'))
b=s.index('    return out, stream2, ids',a)
s=s[:a]+'''    stream1,h=_dense_boundary(o_in,data['stream'],st,'o',st['dense_fusion_choices'].get(T,'library'))
    front=st['dense_front_choices'].get(T,'merged_cublas' if not IS_HOPPER else 'separate')
    pooled,ids,probs,e,pos,counts=_dense_front(h,data['local_start'],st,front)
    back=st['dense_back_choices'].get(T,'serial')
    if T<64:back='serial'
    stream2,out=_dense_back(pooled,probs,e,pos,counts,stream1,st,back)
'''+s[b:]
s=s.replace("json.dumps(st.pop('_measurements'),separators=(',',':'))","json.dumps(_score_compact_table(st.pop('_measurements')),separators=(',',':'))")
s+='\n'+(P/'dev/dense_fusion_paths.py').read_text()+'\n'+(P/'dev/dense_tma_paths.py').read_text()
ast.parse(s)
assert hashlib.sha256(source.read_bytes()).hexdigest()==queued_hash
out=P/'cand_dense_fusion_portfolio.py';out.write_text(s)
print('Built',out,len(s),hashlib.sha256(s.encode()).hexdigest(),'parent',queued_hash)
