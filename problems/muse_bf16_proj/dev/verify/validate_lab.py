"""Force the complete H200 lab glue and its PDL chain on GPU0.

The native portable reader is tested separately. Here a strict ABI stand-in
reaches every Hopper tile/ring family, both output dtypes, exception rejection,
finalists, and both PDL launch choices. Timings are intentionally not reported
as kernel performance.
"""
import importlib.util
import argparse
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
parser=argparse.ArgumentParser();parser.add_argument('--grouped',action='store_true');parser.add_argument('--glue-only',action='store_true')
args=parser.parse_args()
payload='verify_proj_tma_v2.py' if args.grouped else 'verify_proj_tma.py'
spec = importlib.util.spec_from_file_location('lab_tma', ROOT/'prof/lab'/payload)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class Portable:
    def __init__(self):
        self.families = set()
        self.pdl = set()
        self.calls = 0
        self.inject = False

    def wg_gemm(self, w, x, out, n, bt, ns, ncw, st, ks, mc, waves, pipe, pdl):
        assert n == w.shape[0] and out.shape == (x.shape[0], n)
        assert (bt, ns, ncw, st, mc) in (lab.VG_INSTANCES if args.grouped else lab.VP_TMA_INSTANCES)
        assert mc*ks <= 8 and waves in (0, 1, 2) and pipe == 1
        assert isinstance(pdl, bool)
        self.families.add((bt, ns, ncw, st, mc))
        self.pdl.add(pdl)
        self.calls += 1
        if self.inject:
            self.inject = False
            raise AttributeError('intentional missing-family rejection')
        torch.mm(x, w.T, out=out,
                 **({'out_dtype': torch.float32} if out.dtype == torch.float32 else {}))


class PortableGrouped(Portable):
    def wg_gemm(self,w,x,out,n,bt,ns,ncw,st,ks,mc,waves,pipe,pdl,kgroup):
        assert kgroup in (1,2,4) and (w.shape[1]//64)%(ks*kgroup)==0
        super().wg_gemm(w,x,out,n,bt,ns,ncw,st,ks,mc,waves,pipe,pdl)
        self.families.discard((bt,ns,ncw,st,mc))
        self.families.add((bt,ns,ncw,st,mc,kgroup))


def main():
    dev = 'cuda'
    weights = {name: torch.randn(n, k, device=dev, dtype=dtype) for name, n, k, dtype in (
        ('qkv_gate', 2304, 8192, torch.bfloat16),
        ('o', 8192, 1024, torch.bfloat16),
        ('pooler', 4096, 8192, torch.bfloat16),
        ('unpooler', 8192, 4096, torch.bfloat16),
        ('router', 256, 8192, torch.float32))}
    fake = PortableGrouped() if args.grouped else Portable()
    if args.grouped:
        lab.vp_tma_module=lab.vg_tma_module
        lab.vp_tma_lab=lab.vg_tma_lab
    original_module, original_pair, original_unique = lab.vp_tma_module, lab.vp_pair, lab.vp_unique
    # Every specialization is reached at a real decode shape; native validation
    # already covers all K partitions/persistent grids and both dtypes.
    def every_family(m, n, k, sms):
        seen = set()
        answer = []
        for c in original_unique(m, n, k, 132):
            family = c[:5]+(c[-1],) if args.grouped else c[:5]
            if family not in seen:
                seen.add(family)
                answer.append(c)
        return answer

    pairs = []
    def functional_pairs(old, new, x, banks, flush, repeats=20):
        pairs.append(repeats)
        x.copy_(banks[len(pairs) % len(banks)])
        old.graph.replay()
        new.graph.replay()
        torch.cuda.synchronize()
        lab.vp_check(new.out, old.out)
        # This mock controls selection only; none of these values leave the test.
        return 1., 1., 1.

    try:
        lab.vp_tma_module = lambda: fake
        if args.grouped:lab.vg_tma_module=lambda:fake
        lab.vp_pair = functional_pairs
        lab.vp_unique = every_family
        fake.inject = True
        rows = lab.vp_tma_lab(time.monotonic()+240, weights)
        assert any('AttributeError:intentional' in r for r in rows), rows
        prefix='VG' if args.grouped else 'VP'
        assert sum(r.startswith(prefix+'_TMA_BEST ') for r in rows) == 8, rows
        assert sum(r.startswith(prefix+'_CHAIN ') for r in rows) == 4, rows
        assert not any('BUDGET' in r or 'CHAIN_REJECT' in r for r in rows), rows
        expected={c[:5]+(c[-1],) for c in original_unique(256,4608,8192,132)} if args.grouped else set(lab.VP_TMA_INSTANCES)
        assert fake.families == expected, (fake.families,expected)
        assert fake.pdl == {False, True}
        assert 4 in pairs and 20 in pairs
        assert lab._PgGraph.active == 0
        print('VP_FORCED_LAB', dict(families=len(fake.families), calls=fake.calls,
              chains=4, rejected_attribute_error=True, output_characters=sum(len(r)+1 for r in rows)), flush=True)
    finally:
        lab.vp_tma_module, lab.vp_pair, lab.vp_unique = original_module, original_pair, original_unique
        if args.grouped:lab.vg_tma_module=original_module
    if args.glue_only:return
    # Exercise actual native PDL launches and the real boundary arithmetic.
    native = lab.vp_tma_module()
    hi = weights['router'].bfloat16()
    lo = (weights['router']-hi.float()).bfloat16()
    wp = torch.cat((weights['pooler'], hi, lo), 0).contiguous()
    co = (64, 1, 1, 4, 1, 1, 0, 1)
    cp = (64, 1, 1, 4, 2, 2, 1, 1)
    if args.grouped:
        co=(64,1,1,3,1,1,0,1,2);cp=(32,1,1,2,2,2,1,1,4)
    for m in (128, 256):
        x = torch.randn(m, 1024, device=dev, dtype=torch.bfloat16)
        stream = torch.randn(m, 8192, device=dev)
        alpha = torch.full((8192,), .8, device=dev)
        beta = torch.full_like(alpha, .5)
        pre = torch.ones_like(alpha)
        with lab._PgGraph(lambda:lab.vp_chain(x,stream,weights['o'],wp,alpha,beta,pre,None,None,False,native)) as control:
            for pdl in (False, True):
                with lab._PgGraph(lambda:lab.vp_chain(x,stream,weights['o'],wp,alpha,beta,pre,co,cp,pdl,native)) as new:
                    for _ in range(3):
                        x.normal_();stream.normal_()
                        control.graph.replay();new.graph.replay()
                        lab.vp_check(new.out,control.out)
                print('VP_NATIVE_CHAIN', m, pdl, 'fresh3 pass', flush=True)


if __name__ == '__main__':
    main()
