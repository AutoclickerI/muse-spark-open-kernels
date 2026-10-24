"""Check one-stage V9 and vendor grouped SM budgets on GPU0."""
from pathlib import Path
import json
import torch
import deep_gemm as dg
import check_engine as check


def main():
    b = check.base
    w = check.ref.generate_weights(5353)
    native = b.v9_prepare({**w, 'post_norm_w': w['post_expert_norm']})
    bank = b._bf_make_bank(w)
    bank['expected'] = 64
    original_sms = dg.get_num_sms()
    rows = []
    for m in (128, 256):
        datasets = [check.moe_tune.make_fixture(w, m, 5131 + m + 7919*j) for j in range(3)]
        expected = [check.oracle(d, w) for d in datasets]
        for sms in (32, 64):
            for engine in ('masked', 'psum'):
                bank['dg'] = dg
                policy = (engine, 32, .5, 64, sms, 1)
                for d, want in zip(datasets, expected):
                    got = b._bf_hybrid(d, native, bank, policy).float()
                    rel = float((got-want).norm()/want.norm().clamp_min(1e-15))
                    assert torch.isfinite(got).all() and rel < .01, (m,sms,engine,rel)
                    assert dg.get_num_sms() == original_sms
                    rows.append(dict(m=m,sms=sms,engine=engine,l2=rel))
                d = datasets[0]
                for _ in range(2):b._bf_hybrid(d,native,bank,policy)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):out = b._bf_hybrid(d,native,bank,policy)
                for fresh,want in zip(datasets[1:],expected[1:]):
                    for key in ('hidden_states','topk_ids','topk_weights'):d[key].copy_(fresh[key])
                    g.replay();torch.cuda.synchronize()
                    rel=float((out.float()-want).norm()/want.norm().clamp_min(1e-15))
                    assert torch.isfinite(out).all() and rel<.01,('graph',m,sms,engine,rel)
                g.reset()
                # Restore this fixture after refreshed graph replay.
                fresh=check.moe_tune.make_fixture(w,m,5131+m)
                for key in ('hidden_states','topk_ids','topk_weights'):d[key].copy_(fresh[key])
    Path('runs/screen/bf16-resources.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('RESOURCES PASS: 24 eager cases,16 refreshed graph replays; global SM setting restored')


if __name__ == '__main__':main()
