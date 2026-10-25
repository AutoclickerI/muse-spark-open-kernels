"""Least-completion model of shared decoded-B and register-A reuse."""
import hashlib
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
from copy_sync_check import check as copy_check

def check(ng,steps=64,wait=1):
 done=-1;af=[-1,-1];b=[-1,-1];checks=0
 for k in range(steps):
  slot=k&1
  assert done>=b[slot],('live B overwrite',ng,k,done,b)
  # CTA barrier includes every consumer WG's completion frontier, then all
  # decode stores publish before the second CTA barrier and async proxy fence.
  for g in range(ng):
   group=k*ng+g;a=group&1
   assert done>=af[a],('live A overwrite',ng,k,g,done,af)
   af[a]=group;b[slot]=group;done=max(done,group-wait);checks+=2
 done=steps*ng-1
 assert all(done>=v for v in b+af)
 return checks

def negative():
 try:check(2,wait=2)
 except AssertionError as e:return str(e)
 raise AssertionError('missing negative control')

cases=[dict(ng=n,k=k,checks=check(n,k)) for n in (1,2,4) for k in (2,3,63,64,65)]
# Bijective fragment maps: each BF16 element belongs to exactly one pair store.
for nc in (64,128):
 pairs=[]
 for wg in range(nc//64):
  for t in range(128):
   w,l=divmod(t,32)
   for j in range(16):
    r=wg*64+w*16+(l>>2)+(j&1)*8;k=(l&3)*2+((j>>1)&1)*8+(j>>2)*16
    pairs.extend([(r,k),(r,k+1)])
 assert len(pairs)==nc*64 and len(set(pairs))==len(pairs)
# W13's uint2 stores form the entire W2 A-fragment tensor at each channel tile.
for nc in (64,128):
 cells=[]
 for t in range(128):
  w,l=divmod(t,32)
  for i in range(0,nc//8,2):
   ch=(i//2)*8+(l&3)*2;at=w*32+(l>>2)*4+(l&3);word=2 if ch&8 else 0
   cells.extend([(ch//16,at,word),(ch//16,at,word+1)])
 assert len(cells)==nc//2*64//2 and len(set(cells))==len(cells)
# Code words and scale bytes are fetched exactly once per output/K tile.
# Only the first lane in each quartet loads scales; SHFL serves its peers.
weight_counts={}
for nc in (64,128):
 qwords=[];scbytes=[]
 for wg in range(nc//64):
  for t in range(128):
   w,l=divmod(t,32)
   for kh in range(64):
    base=((wg*4+w)*128+kh*2+(l>>4))*64+(l&15)*4
    qwords.extend(range(base,base+4))
    if l%4==0:
     base=((wg*4+w)*32+kh//2)*128+(l>>2)*16+(kh&1)*8
     scbytes.extend(range(base,base+8))
 assert len(qwords)==nc*4096//8 and len(set(qwords))==len(qwords)
 assert len(scbytes)==nc*4096//16 and len(set(scbytes))==len(scbytes)
 weight_counts[nc]=dict(Q_u32=len(qwords),scale_u8=len(scbytes),multiplicity=1)
copy_checks=sum(copy_check(s,k) for s in (1,2,3) for k in range(3,66))
out=dict(copy_ring_checks=copy_checks,checks=sum(x['checks'] for x in cases),cases=cases,negative_wait2=negative(),B_and_A2_layout_bijective=True,packed_weight_source_counts=weight_counts,cuda_sha256=hashlib.sha256((H/'transpose.cu').read_bytes()).hexdigest(),limitation='Contract/layout model, not a hardware race test. Matching sm120 tensor-core execution checked separately.')
(H/'sync.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
