"""Reproduce the source-count audit and capacity thresholds (decimal GB).

Run in serving/harness/img with the patch on PYTHONPATH. Dynamic H200 pools
remain an explicit unknown; a full-width local graph proxy is not their total.
"""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report():
    spec = importlib.util.spec_from_file_location('weight_count', ROOT / 'serving/tests/muse_residency.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = mod.count()
    r = base['fixed_components_bytes']
    budget = base['static_budget_bytes']
    core_patch = base['fixed_total_bytes']
    # Marlin stores the processed scales as FP8 again, not BF16.
    core_stock = core_patch + (5 * 8192 * 8192 * 2) * (1 - 1 / 8) - 60 * 32 * 8
    receipt_layouts = 62 * (4096 * 8192 + 8192 * 1024) * 2
    optional_layouts = 62 * (8192 * 4096 + 512 * 8192 + 1024 * 4096) * 2
    old_router = 62 * 40 * (4 * 17 * 64 * 64 * 4 + 4 * 4)
    shared_router = 16 * 17 * 64 * 64 * 4 + 16 * 4
    # Muse exposes 2047 previous positions to the cache (2048 including the
    # current position); page-size-one padding supplies that final position.
    swa_rows, full_cell, swa_cell = 262657, 5376, 11776
    static = dict(stock=core_stock,
                  patch_before=core_patch + receipt_layouts + old_router,
                  patch_shared_router=core_patch + receipt_layouts + shared_router,
                  patch_shared_router_max_layouts=core_patch + receipt_layouts + optional_layouts + shared_router)
    scenarios = []
    for c in (1, 4, 16, 32):
        for length in (80000, 131072):
            rows = c * length
            default_bytes = rows * (4096 + .8 * 11776 + 1280)
            bounded_bytes = rows * full_cell + swa_rows * swa_cell
            scenarios.append(dict(concurrency=c, context=length,
                                  full_rows=rows, default_split_kv_bytes=int(default_bytes),
                                  bounded_split_kv_bytes=bounded_bytes,
                                  max_other_resident_bytes_before_reclaim=budget-static['patch_before']-default_bytes,
                                  max_other_resident_bytes_after_reclaim=budget-static['patch_shared_router']-bounded_bytes,
                                  max_other_resident_bytes_max_layouts=budget-static['patch_shared_router_max_layouts']-bounded_bytes))
    return dict(basis=base['basis'], static_components_patch_bytes=r,
                static_budget_bytes=budget, static_totals_bytes=static,
                receipt_layouts_bytes=receipt_layouts,
                optional_selected_layouts_bytes=optional_layouts,
                humming_conditional_bytes=905969664,
                router_before_bytes=old_router, router_shared_stream_bytes=shared_router,
                bounded_swa_rows=swa_rows, bounded_swa_bytes=swa_rows*swa_cell,
                full_cell_target_and_draft_bytes=full_cell,
                default_cell_target_and_draft_bytes=14796.8,
                max_full_rows_equations={
                    'stock': '(119850000000 - static_stock - U_stock) / 14796.8',
                    'patch_before': '(119850000000 - static_patch_before - U_patch) / 14796.8',
                    'patch_bounded_shared_router': '(119850000000 - static_patch_shared - U_patch - 3093048832) / 5376'},
                scenarios=scenarios,
                unknown='U includes actual H200 graph pools, graph I/O, KV materialization, request maps, allocator slack and CUDA/NCCL allocations; conditional layouts/bank must also be charged. Source defaults for vision/audio and RoPE need checkpoint confirmation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for key, value in result['static_totals_bytes'].items():
        print(f'{key}: {value / 1e9:.6f} GB')
    for row in result['scenarios']:
        print(f"c{row['concurrency']:2} L{row['context']:6}: default {row['default_split_kv_bytes']/1e9:.6f}, bounded {row['bounded_split_kv_bytes']/1e9:.6f}, other-resident allowance {row['max_other_resident_bytes_after_reclaim']/1e9:.6f} GB")
