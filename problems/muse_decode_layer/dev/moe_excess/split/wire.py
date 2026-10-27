"""Run a split payload end to end on the local GPU (sm_120 stand-ins for WGMMA) and size its packet.

Usage: CUDA_VISIBLE_DEVICES=0 tools/img python .../split/wire.py J|I [--seeds N]
Local timing is not evidence; this proves every arm executes eager and in graphs, the TASK gates,
the reader round trip and the public packet size.
"""
import importlib.util, json, sys, time
from pathlib import Path
import torch
H = Path(__file__).resolve().parent; R = H.parents[4]; sys.path.insert(0, str(R / 'tools'))
from lab_packet import encode_packet
name = sys.argv[1]; seeds = int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else 8
spec = importlib.util.spec_from_file_location('split_wire', H / f'out/split_{name}.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
flush = torch.empty(1 << 30, dtype=torch.uint8, device='cuda')
m.lab_cold_l2 = lambda: flush.view(torch.int64).sum()
m.SEEDS = m.SEEDS[:seeds]
m.XL_LOCAL_DUMP = lambda d: (H / f'out/local_raw_{name}.json').write_text(json.dumps(d))
t = time.monotonic(); out = m.lab(time.monotonic() + 3600, None)
packet = encode_packet('moe_split_' + name, m.sha((H / f'out/split_{name}.py').read_text())[:16], dict(lines=out, complete=None, format='legacy-lines'))
n = len(packet.encode()) + 1
(H / f'out/local_wire_{name}.txt').write_text('\n'.join(out))
print('WIRE', name, n, 'seeds', seeds, 'elapsed', round(time.monotonic() - t, 1))
assert n <= 11744, n
