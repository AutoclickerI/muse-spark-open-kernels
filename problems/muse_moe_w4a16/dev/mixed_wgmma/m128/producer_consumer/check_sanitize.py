"""Focused graph replay for sanitizer inspection of the two warpgroup rings."""
import importlib.util
import argparse
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[6]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cfg',type=int,choices=(1,3));ap.add_argument('--eager-only',action='store_true');args=ap.parse_args()
    s=importlib.util.spec_from_file_location('pc_sanitize',ROOT/'prof/lab/moe_m128_producer_consumer.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    w=m.dc_weights(5353,'cuda');w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs'];bank=m.hh_prepare(w)
    d=m.dc_fixtures(w,128)[0];ids=torch.full_like(d['topk_ids'],255);ids[:80,0]=0;d=dict(d,topk_ids=ids)
    for cfg in ((args.cfg,) if args.cfg is not None else (1,3)):
        plan=m.pc_plan(d,bank,cfg,132,3,16);steps=plan['steps']
        for f in steps:out=f()
        if not args.eager_only:
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in steps:out=f()
            g.replay()
        torch.cuda.synchronize();assert bool(torch.isfinite(out).all())
    print('PC_SANITIZE_GRAPH_PASS',flush=True)


if __name__=='__main__':main()
