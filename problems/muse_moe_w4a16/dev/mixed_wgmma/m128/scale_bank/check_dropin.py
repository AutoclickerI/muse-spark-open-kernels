"""Gate the standalone cell with signed/zero/subnormal scales and changing routes."""
import importlib.util
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    control = load('m128_scale_control', ROOT / 'prof/lab/moe_m128_scale_bank.py')
    cell = load('m128_scale_cell', HERE / 'dropin.py')
    w = control.dc_weights(38471, 'cuda')
    for key in ('w13_sf', 'w2_sf'):
        raw = w[key].view(torch.uint8)
        raw[..., ::3] ^= 128
        raw[..., 1::37] = 0
        raw[..., 2::41] = 1
    bank = cell.m128_scales_scale_prepare(w)
    ds = control.dc_fixtures(w, 128)
    for kind in ('empty', 'single', 'hot', 'balanced'):
        ids = torch.full_like(ds[0]['topk_ids'], 255)
        if kind == 'single': ids[3, 2] = 31
        if kind == 'hot': ids[:, 0] = 0
        if kind == 'balanced': ids[:, 0] = torch.arange(128, device='cuda') % 32
        ds.append(dict(ds[0], hidden_states=torch.randn_like(ds[0]['hidden_states']), topk_ids=ids))
    wants = control.dc_oracles(ds, w); results = []
    for cfg in (3, 4, 5):
        graph = control.MWGraph(lambda d: cell.m128_scales_scale_call(d, bank, cfg, 528), ds[0])
        for i, (data, want) in enumerate(zip(ds, wants)):
            graph.refresh(data); got = graph.replay(); torch.cuda.synchronize()
            error, row = control.scale_error(got, want)
            assert error <= .01 and row <= .05, (cfg, i, error, row)
            results.append(dict(config=cfg, fixture=i, L2=error, row=row))
    report = dict(checks=len(results), signed_zero_subnormal_scales=True, separate_gate_up_global_scales=True,
        routes=['real0', 'real1', 'real2', 'empty', 'single', 'hot', 'balanced'],
        max_L2=max(r['L2'] for r in results), max_row=max(r['row'] for r in results), results=results)
    (HERE / 'dropin_agreement.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SCALE_DROPIN_PASS', report['checks'], report['max_L2'], report['max_row'], flush=True)


if __name__ == '__main__':
    main()
