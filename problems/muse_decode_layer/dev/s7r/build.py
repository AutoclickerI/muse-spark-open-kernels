"""S7R: S7 with the M=256 MoE on R_full (MoE-host lab 110599ba6b6144d4).

mixed_parallel runs route128 (zero-fills the output and readiness state), W13 light||heavy, then one rc W2.
R_full keeps S6's buffers and recoded bank and replaces those steps: a one-CTA live route (no fills), then one
fork: side = every expert's <=16-row 32-row groups on N16, W13 then W2; main = its full groups on N32, W13 then
W2; joined before the same distributed finalizer. 320.61 vs 347.13 us, paired 0.9227 on 12 seeds x3 fixtures,
bit-equal to mixed_parallel on all 36. Kernels: problems/muse_moe_w4a16/dev/mixed_wgmma/branches/ (KINDS_S6).
"""
from pathlib import Path
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
BASE = R / 'problems/muse_decode_layer/cand_r3_best7_s7.py'
BASE_SHA = 'bd3932d1cc3c0c82d60145772148fd19fc779465dd360bda610d214385401390'
sys.path[:0] = [str(R / 'problems/muse_moe_w4a16/dev/mixed_wgmma/branches'), str(R / 'tools')]
import sources  # noqa: E402
from okc_run import rejected_words  # noqa: E402

CELL = '''

# MoE lane H200 110599ba6b6144d4: M256 R_full on mixed_parallel's buffers (bit-equal; paired 0.9227).
RF_CUDA={cuda!r}
RF_CPP={cpp!r}
@functools.lru_cache(None)
def rf_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='rf_m256_'+hashlib.sha256(RF_CUDA.encode()).hexdigest()[:12],cpp_sources=RF_CPP,cuda_sources=RF_CUDA,
        functions=['br_init','br_run','br_route','br_finish'],no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.br_init()
    return mod

def rf_plan(data,bank):
    plan=wi_plan(data,bank,6,2112);mod=rf_module()
    v13,v2=plan['v13'],plan['v2'];m,start=plan['M'],plan['start']
    q13=[bank['w13_rec'],bank['w13_rec_sf']]+[v13[i] for i in (2,3,4,5,6,7,8,14,9,15,16)]
    q2=[bank['w2_rec'],bank['w2_rec_sf']]+[v2[i] for i in (2,3,4,5,6,7,8,14,9,15,16)]
    p13=[t.data_ptr() for t in q13];p2=[t.data_ptr() for t in q2]
    rp=[v2[i].data_ptr() for i in (11,4,5,6,7,8,13,7,7)];fp=[v2[i].data_ptr() for i in (5,11,12,13,15,16,19)]
    owners=(q13,q2,v13,v2);side,fork,join=bank['mix_stream'],bank['mix_start'],bank['mix_done']
    def route():
        _=owners;mod.br_route(rp,m,start,0,torch.cuda.current_stream().cuda_stream)
    def body():
        _=owners;parent=torch.cuda.current_stream()
        fork.record(parent);side.wait_event(fork)
        with torch.cuda.stream(side):
            mod.br_run(p13,5,2112,m,side.cuda_stream);mod.br_run(p2,9,2112,m,side.cuda_stream);join.record()
        mod.br_run(p13,4,2112,m,parent.cuda_stream);mod.br_run(p2,8,2112,m,parent.cuda_stream)
        parent.wait_event(join)
    def finish():
        _=owners;mod.br_finish(fp,m,start,torch.cuda.current_stream().cuda_stream);return v2[19]
    plan['steps']=(route,body,finish)
    return plan

def rf_call(data,bank):
    for f in rf_plan(data,bank)['steps']:out=f()
    return out
'''
ANCHOR = '''def wi_call(data, bank, policy=-1, grid=2112):
    for f in wi_plan(data, bank, policy, grid)['steps']:
        out = f()
    return out
'''
PY_EDITS = (
    ("    if name=='mixed_parallel_m256':return wi_call(data,state['s6_bank'],6,2112)",
     "    if name=='mixed_parallel_m256':return rf_call(data,state['s6_bank'])", 1),
    ("    graph=_best5_graph(lambda data:wi_call(data,state['s6_bank'],6,2112),fixtures[0])",
     "    graph=_best5_graph(lambda data:rf_call(data,state['s6_bank']),fixtures[0])", 1),
    ("bank='lossless recoded W13 and W2',schedule='parallel light/heavy W13; joined before W2')",
     "bank='lossless recoded W13 and W2',schedule='R_full: <=16-row groups on N16 W13->W2 beside N32 groups, forked once, joined before the finalizer')", 1),
)


def main():
    s = BASE.read_text()
    assert hashlib.sha256(s.encode()).hexdigest() == BASE_SHA
    src = sources.sources()
    assert s.count(ANCHOR) == 1
    s = s.replace(ANCHOR, ANCHOR + CELL.format(cuda=src['cu_s6'], cpp=src['cpp_s6']))
    for old, new, n in PY_EDITS:
        assert s.count(old) == n, (old, s.count(old))
        s = s.replace(old, new)
    assert 'wi_call(data,state' not in s
    assert not rejected_words(s), rejected_words(s)
    compile(s, 'cand', 'exec')
    out = BASE.with_name('cand_r3_best7_s7r.py')
    out.write_text(s)
    m = dict(source=str(out.relative_to(R)), sha256=hashlib.sha256(s.encode()).hexdigest(), base=str(BASE.relative_to(R)), base_sha256=BASE_SHA,
             rf_cuda_sha256=hashlib.sha256(src['cu_s6'].encode()).hexdigest(), lab='110599ba6b6144d4 R_full 0.9227 vs mixed_parallel, bit-equal',
             change='M256 MoE: live route + fork-once N16 remainder groups || N32 full groups, both phases; same finalizer and bank')
    (H / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    print(m['source'], m['sha256'])


if __name__ == '__main__':
    main()
