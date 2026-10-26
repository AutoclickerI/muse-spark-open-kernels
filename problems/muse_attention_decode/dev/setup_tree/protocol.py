"""CPU proof of quad-owned page-coordinate coverage for every copied sector."""
import json,random
from pathlib import Path
H=Path(__file__).resolve().parent
def main():
    rng=random.Random(71);count=0;sectors=0
    for cs in (4,8):
      for window in (1,63,64,511,1024,2048):
       for seq in (0,1,10,2047,2048,80000):
        for ps in (1,64):
         lo=max(0,seq-window+1);nc=seq-lo;chunk=((nc+cs-1)//cs+127)//128*128
         pt=[rng.randrange(100000) for _ in range(max(1,(seq+ps-1)//ps))]
         for split in range(cs):
          begin=min(seq,lo+split*chunk);end=min(seq,begin+chunk);nt=(end-begin+127)//128
          assert nt<=4
          for tile in range(nt):
           for tid in range(128):
            lane=tid&31;owner=(tid//32)*32+(lane&~3)+(tile&3)
            for j in range(4):
             key=begin+tile*128+tid//4+32*j;clamped=min(key,max(seq-1,0))
             old=pt[clamped//ps]*ps+clamped%ps
             prefkey=begin+(owner&3)*128+owner//4+32*j;prefclamp=min(prefkey,max(seq-1,0))
             new=pt[prefclamp//ps]*ps+prefclamp%ps
             assert old==new and key==prefkey;sectors+=1
         count+=1
    out=dict(cases=count,copied_sectors_proved=sectors,page_sizes=[1,64],native_cs=[4,8],max_native_tiles=4,
             proof='For each copy sector, quad owner has identical logical key and physical page/row; duplicates become shuffles. Native windows<=2048 imply<=4 tiles per selected split. Local C1 extends cache to16 tiles; no native arithmetic is inferred from local time.')
    (H/'protocol.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
