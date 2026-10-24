"""Resident TP2 controls, preserving each server's full-grid cache history."""

import argparse
import json
from pathlib import Path

import persistent_ttft_brackets as b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    out = b.DATA / 'results' / f'{args.prefix}-evidence'
    out.mkdir(exist_ok=False)
    results = []
    # Separate mode phases keep three resident models, leaving GPU1 memory
    # for its other project. Both adjacent controls retain independent caches.
    for mode in ('native', 'portable'):
        roles = ('stock-a', mode, 'stock-b')
        servers = {}
        try:
            for role in roles:
                tree = b.run.common.STOCK_TREE if role.startswith('stock') else args.tree
                name = f'{role}-{mode}' if role.startswith('stock') else role
                server = b.launch(name, 2, tree, args.prefix)
                servers[role] = server
                b.workload(server, f'{args.prefix}-warm-{name}-tp2', 1, 2, 3)
                b.flush(server)
            # Each server advances through c1/c4/c16/c32 exactly once, as in
            # the original full-grid harness. No flush/probe between levels.
            for level, convs in ((1, 2), (4, 8), (16, 16), (32, 32)):
                labels = {}
                for role in roles:
                    label = f'{args.prefix}-c{level}-{role}-{mode}-tp2'
                    print('BRACKET', label, flush=True)
                    labels[role] = b.workload(servers[role], label, level, convs, 3, probe=level == 32)
                for control in ('stock-a', 'stock-b'):
                    result = b.paired(labels[control], labels[mode], 2, level, mode)
                    print(json.dumps(result), flush=True)
                    results.append(result)
                (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        finally:
            for server in servers.values():
                b.run.stop(server['proc'])
    assert all(row['margin_pass'] for row in results), 'full-history p50/p90 margin failed; evidence retained'
    print('FULL HISTORY AGREEMENT AND P50/P90 MARGIN PASS', flush=True)


if __name__ == '__main__':
    main()
