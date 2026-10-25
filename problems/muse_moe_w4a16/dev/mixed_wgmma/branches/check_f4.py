"""F4 vs S11 final, exact same banks and inputs, including captured replay.

The native finalizer is checked separately on synthetic Y/SS so the known NT1
norm atomic replay nondeterminism cannot hide a finalizer difference.
"""
import hashlib
import importlib.util
import json
import runpy
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
R = H.parents[4]


def main():
    path = H / 's11_f4.py'
    spec = importlib.util.spec_from_file_location('f4', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    lab = runpy.run_path(str(R / 'prof/lab/moe_nf_tune_m128.py'))
    mod = m.nf_module()
    g = torch.Generator(device='cuda').manual_seed(5353)
    rows = []
    for M in (128, 256):
        w = lab['dc_weights'](5353, 'cuda')
        wp = dict(w, post_norm_w=w['post_expert_norm'])
        base = m.issue_prepare(m.nf_ilv(wp))
        native = m._prepare_recoded_bank(w, m.v9_prepare(wp), m._v9_recoded_module())
        streams = (torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event())
        for data in lab['dc_fixtures'](w, M):
            p = m.nf_plan(data, base, native, streams)
            for fn in p['steps']: fn()
            torch.cuda.synchronize()
            # Same Y and SS from one producer execution, compare split 1 and 4.
            h13, h2, nv, ids, tw, pos, tasks = p['owners']
            out1, out4 = torch.empty_like(p['out']), torch.empty_like(p['out'])
            args = [h13[5], ids, tw, pos, h13[11], h13[12]]
            for split, out in ((1, out1), (4, out4)):
                mod.br_finish([t.data_ptr() for t in args + [out]], M, 0, split, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            assert torch.equal(out1, out4)
            rows.append(dict(M=M, source='S11 producer', bit_equal=True))
        # Arbitrary finite scratch, all local routes and nonzero local_start.
        cap = M * 8
        off = torch.arange(33, device='cuda', dtype=torch.int32) * M
        ids = torch.arange(8, device='cuda', dtype=torch.int32).expand(M, 8).contiguous() + 64
        pos = torch.arange(M, device='cuda', dtype=torch.int32)[:, None].expand(M, 8).contiguous()
        tw = torch.rand((M, 8), device='cuda', generator=g)
        y = torch.randn((cap, 4096), device='cuda', generator=g)
        ss = torch.rand((cap, 64), device='cuda', generator=g) * 64
        out1, out4 = torch.empty((M, 4096), device='cuda', dtype=torch.bfloat16), torch.empty((M, 4096), device='cuda', dtype=torch.bfloat16)
        args = [off, ids, tw, pos, y, ss]
        for split, out in ((1, out1), (4, out4)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mod.br_finish([t.data_ptr() for t in args + [out]], M, 64, split, torch.cuda.current_stream().cuda_stream)
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out1, out4)
        rows.append(dict(M=M, source='synthetic captured finalizer', bit_equal=True))
    result = dict(candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows)
    (H / 'f4_bitexact.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
