"""Adversarial progress/ownership model for TMA, decode and two async consumers.

Models pending WGMMA X reads rather than assuming that issuing an MMA releases X.
This is a protocol check, not a proof of compiler/hardware memory ordering.
"""
import hashlib
import json
from pathlib import Path
import random
H=Path(__file__).resolve().parent

def trial(slots,pipe,seed,bad=None):
 rng=random.Random(seed);jobs=3;total=jobs*32;clock=0;producer=decoder=0
 raw={};x={};a={};raw_free=set(range(-2,0));empty={i:{0,1} for i in range(-slots,0)}
 raw_full=set();x_full=set();releases={};pending=[[],[]];cp=[0,0];sub=[0,0];ep_done=[-1,-1];events=[];overlap=False
 def release(c,k):
  assert k>=0 and c not in releases.setdefault(k,set()),('double release',c,k)
  releases[k].add(c);empty[k]=releases[k]
 for steps in range(30000):
  ready=[e for e in events if e[0]<=clock]
  for e in ready:
   events.remove(e);_,kind,k=e
   if kind=='r':raw_full.add(k)
   elif kind=='x':x_full.add(k)
   else:ep_done[k]=cp[k]//32-1
  if producer==decoder==total and cp==[total,total] and not events:
   assert all(v=={0,1} for k,v in releases.items() if k>=0)
   assert len(releases)==total and not any(pending)
   return overlap
  actions=[]
  if producer<total and producer-2 in raw_free and empty.get(producer-slots)=={0,1} and (pipe or min(ep_done)>=producer//32-1):actions.append('p')
  if decoder<total and decoder in raw_full and empty.get(decoder-slots)=={0,1} and (pipe or min(ep_done)>=decoder//32-1):actions.append('d')
  for c in (0,1):
   k=cp[c]
   if k<total and ep_done[c]>=k//32-1 and a.get(k%slots)==k and k in x_full:actions.append(c)
  if not actions:
   if events:clock=min(e[0] for e in events);continue
   raise AssertionError(('deadlock',slots,pipe,seed,producer,decoder,cp,sub))
  who=rng.choice(actions);clock+=1
  if who=='p':
   k=producer
   assert all(k-slots not in q for q in pending),('overwrite pending X',k,pending)
   assert raw.get(k%2,k-2) in raw_free
   raw[k%2]=k;x[k%slots]=k
   events.extend([(clock+rng.randrange(1,10),'r',k),(clock+rng.randrange(1,15),'x',k)])
   if k%32==0 and k and min(ep_done)<k//32-1:overlap=True
   producer+=1
  elif who=='d':
   k=decoder;assert raw[k%2]==k
   a[k%slots]=k;raw_free.add(k);decoder+=1
  else:
   c=who;k=cp[c];assert x[k%slots]==a[k%slots]==k
   pending[c].append(k)
   if len(pending[c])>1:pending[c].pop(0)  # wait_group1
   if bad=='early' and sub[c]==0:release(c,k)
   elif slots>1 and sub[c]==0 and k%32:release(c,k-1)
   if sub[c]==0:sub[c]=1;continue
   sub[c]=0
   if slots==1 and bad!='lag':pending[c].clear();release(c,k)
   if k%32==31:
    pending[c].clear()
    if slots>1 and bad!='early':release(c,k)
    events.append((clock+50,'e',c))
   cp[c]+=1
 raise AssertionError('step bound')

checks=0;overlaps=0
for slots in (1,2):
 for pipe in (False,True):
  for seed in range(100):
   overlaps+=trial(slots,pipe,seed);checks+=1
negative=[]
for slots,bad in ((1,'lag'),(2,'early')):
 try:trial(slots,True,0,bad)
 except AssertionError as e:negative.append(str(e));checks+=1
 else:raise AssertionError(('bad protocol accepted',bad))
assert overlaps>0
# Same statically owned (expert,two-channel-tile,row-range) list in every role.
for m in (33,128,256,257,1024):
 for first,tiles in ((True,64),(False,32)):
  counts=[min(m,n) for n in (0,1,8,9,16,17,32,33,63,64,65,96,127,128,129,192,224,225,255,256,257,512,1024)]+[0]*9
  for grid in (1,2,132,264):
   found=[]
   for maximum in (64,128,192,256):
    for cta in range(grid):
     for u in range(cta,32*tiles,grid):
      e,mt=divmod(u,tiles);n=counts[e]
      if n<=16:continue
      for row in range(0,n,256):
       rem=min(n-row,256)
       if maximum-64<rem<=maximum:found.append((e,mt,row))
   want={(e,mt,row) for e,n in enumerate(counts) if n>16 for mt in range(tiles) for row in range(0,n,256)}
   assert len(found)==len(set(found)) and set(found)==want
   # Two 64-channel consumers x four warps x32 channel-pairs cover every HP entry.
   hp={((mt*2+consumer)*4+warp) for mt in range(32) for consumer in (0,1) for warp in range(4)}
   assert hp==set(range(256));checks+=1
out=dict(source_sha256=hashlib.sha256((H/'pipeline.cu.inc').read_bytes()).hexdigest(),checks=checks,random_schedules=400,pipe_schedules_with_cross_tile_overlap=overlaps,negative_controls=negative,raw_slots=2,decoded_slots=[1,2],stages_per_tile=32,ownership='Empty barrier requires both consumers. S2 releases the prior stage after the next group0 wait1; final wait0 releases the final stage. S1 drains/releases current. Raw TMA completion precedes decode, decode completion precedes A loads. Next tile may stream before both consumer epilogues finish; gated arm waits for both.',limitation='Models ownership and forward progress under variable completion order. Native TMA proxy visibility, instruction semantics, register reallocation and Hopper two-slot execution require H200.')
(H/'protocol.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
