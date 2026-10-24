"""Refresh the static issue and routing model; these are not hardware stall counters."""
import json
import re
from pathlib import Path
from count import parse
from dataflow import classify

H = Path(__file__).resolve().parent

def row(name, ins):
    c = {k: v / 4 for k, v in classify(ins).items()}
    totals = dict(unpack=c.get('FP4_unpack', 0),
                  scale=sum(v for k, v in c.items() if k.startswith('scale_')),
                  WGMMA=c.get('WGMMA_issue_fence_wait', 0),
                  loads=sum(c.get(k, 0) for k in ('Q_shared_load', 'SF_shared_load', 'async_weight_activation_load')),
                  address=sum(v for k, v in c.items() if k.startswith('address_')),
                  other=sum(c.get(k, 0) for k in ('control_barrier_other', 'predicated_off_fence_scaffold')))
    assert abs(sum(totals.values()) - len(ins) / 4) < 1e-9
    return dict(name=name, total=len(ins), per16=c, summary_per16=totals,
                issue_roof_TBs_1980MHz_132SM=128 * 132 * 1.98e9 / (len(ins) / 64) * 9 / 16 / 1e12)

def main():
    rows = []
    for name in ('raw', 'recoded'):
        ins = []
        for line in (H / (name + '_loop.txt')).read_text().splitlines():
            m = re.match(r'([0-9a-f]+)\s+(@!?\w+)?\s*([A-Za-z][\w.]*)\s+(.*)', line)
            assert m, line
            ins.append((int(m[1], 16), m[3], m[4], m[2] or ''))
        rows.append(row(name, ins))
    for part in (H / 'native.sass').read_text().split('Function : ')[1:]:
        ins = parse(part)
        loops = []
        for pc, op, arg, pred in ins:
            m = re.search(r'0x([0-9a-f]+)', arg)
            if op == 'BRA' and m and int(m[1], 16) < pc:
                lo = int(m[1], 16)
                body = [i for i in ins if lo <= i[0] <= pc]
                if sum(i[1].startswith('HGMMA.') for i in body) == 8:
                    loops.append((pc - lo, body))
        _, body = min(loops)
        rows.append(row(part.splitlines()[0].strip(), body))
    routing = json.loads((H / 'routing.json').read_text())
    counts = routing['counts']
    cold = [n for n in counts if n <= 16]
    hot = [n for n in counts if n > 16]
    padded = len(cold) * 16 + sum((n + 31) // 32 * 32 for n in hot)
    routing.update(cold_experts=len(cold), cold_pairs=sum(cold), hot_experts=len(hot), hot_pairs=sum(hot),
                   mixed_padded_rows=padded, mixed_token_fill=sum(counts) / padded,
                   cold_requested_byte_fraction=len(cold) / routing['groups32'])
    out = dict(rows=rows, routing=routing,
               measured_R0=dict(receipt='97be11d4631b44d6', NT32_W13_us=261.2,
                                requested_TBs=routing['requested_W13_bytes'] / 261.2 / 1e6,
                                unique_TBs=routing['unique_W13_bytes'] / 261.2 / 1e6),
               note='Static SASS per K128 step:64 weights/lane, four16-weight groups. Includes mutually exclusive copy paths and predicated-off fence scaffold; not dynamic instructions. Uniform instructions count as issued warp slots. Issue ceilings assume128 lane-instructions/cycle/SM at1980MHz. Prefetch capacity is nominal, not measured outstanding HBM requests. Shared-load provenance categories are a source-assisted approximation. Scale includes lookup loads; loads excludes those. Requested bytes include repeated expert-row blocks; neither quantity is measured DRAM traffic.')
    (H / 'model.json').write_text(json.dumps(out, indent=2) + '\n')
    for r in rows:
        print(r['name'], r['total'] / 4, r['summary_per16'])

if __name__ == '__main__':
    main()
