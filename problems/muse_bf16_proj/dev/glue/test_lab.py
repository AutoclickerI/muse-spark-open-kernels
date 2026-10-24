"""Execute the complete lab glue on GPU0. CUTLASS/FA3/MoE aren't needed here."""
from pathlib import Path
import importlib.util,time,json,argparse
import torch
p=argparse.ArgumentParser();p.add_argument('--hopper-glue',action='store_true');p.add_argument('--short',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('glue_lab',root/'prof/lab/verify_projection_glue.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
weights={n:(torch.randn(a,b,device='cuda')/(b**.5)).to(dt) for n,a,b,dt in (
    ('qkv_gate',2304,8192,torch.bfloat16),('o',8192,1024,torch.bfloat16),
    ('pooler',4096,8192,torch.bfloat16),('unpooler',8192,4096,torch.bfloat16),
    ('router',256,8192,torch.float32))}
if args.hopper_glue:
    # Lt CUDA is portable and actually executes. Force the conditional through
    # its real image modules/options/shortlist; only architecture metadata differs.
    g._gbc__fa3_on_device=lambda device:True
if args.short:
    original=g.gb_choices
    g.gb_choices=lambda m,p:list(original(m,p))[:2]
rows=g.lab(time.monotonic()+240,weights)
for row in rows:print(row,flush=True)
assert any(r.startswith('GB_LAB_END') for r in rows),rows
assert sum(r.startswith('GB_STAGE') for r in rows)==18,[r for r in rows if 'GB_REJECT' in r or 'GB_BUDGET' in r]
chars=sum(len(r)+1 for r in rows);print('LAB_COMPACT_CHARS',chars,flush=True);assert chars<9000
(root/'runs/screen/verify_proj/glue_lab_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
