"""Route shape per M over public seeds: local experts, pairs, NT1 blocks (8 tokens), unique vs issued weight bytes."""
import linecache,statistics,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text();n={};f='<lab>';linecache.cache[f]=(len(src),None,src.splitlines(True),f)
exec(compile(src,f,'exec'),n)
TILE=36864;EXP=(512+256)*TILE
for m in (1,8,16,32):
    rows=[]
    for seed in range(8):
        w=n['rg_generate_weights'](5353+7919*seed);d=n['fx_make_fixture'](w,m,9000+m+97*seed)
        ids=d['topk_ids'].flatten();ids=ids[ids>=0];cnt=torch.bincount(ids.long(),minlength=32)
        e=int((cnt>0).sum());blk=int(((cnt+7)//8).sum());rows.append((e,len(ids),blk,int(cnt.max())))
        del w
    E,P,B,X=(statistics.mean(r[i] for r in rows) for i in range(4))
    print(f'M{m}: local experts {E:.1f} pairs {P:.1f} blocks {B:.1f} max/expert {X:.1f} unique {E*EXP/1e6:.0f} MB issued {B*EXP/1e6:.0f} MB floor@4.8 {E*EXP/4.8e6:.1f} us @3.6 {E*EXP/3.6e6:.1f} us')
