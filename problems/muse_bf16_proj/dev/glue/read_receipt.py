"""Read a landed glue lab receipt; never query, wait or poll the judge."""
import argparse,ast,hashlib,json,pprint,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path)
p.add_argument('--recipes',type=Path);a=p.parse_args()
doc=json.loads(a.receipt.read_text());tail=doc['result'].get('log_tail','')
sha=hashlib.sha256((ROOT/'prof/lab/verify_projection_glue.py').read_bytes()).hexdigest()[:16]
if f'LAB_BEGIN verify_projection_glue {sha}' not in tail:raise SystemExit('wrong payload SHA; do not attribute this receipt')
stages={};chains={}
for line in tail.splitlines():
 if line.startswith(('GB_META','GB_CONTROL','GB_PREFIX','GB_REJECT','GB_BUDGET','GB_END','GB_LAB_END','LAB_END')):print(line)
 m=re.match(r'GB_STAGE (\d+) (b1|pool|b2) choice=(.*?) ratio=([^ ]+) (.*)',line)
 if m:
  tokens,phase,c,ratio,rest=m.groups();c=ast.literal_eval(c)
  print(f'{tokens} {phase}: best_native={c} r={ratio} {rest}')
  if c is not None:stages[int(tokens),phase]=c
 m=re.match(r'GB_CHAIN (\d+) enabled=(True|False) choice=(.*?) ratio=([^ ]+) all=(.*)',line)
 if m:
  tokens,enabled,choice,ratio,all_rows=m.groups();choice=ast.literal_eval(choice)
  print(f'CHAIN M={tokens} enabled={enabled} choice={choice} r={ratio} us20={all_rows}')
  if enabled=='True' and choice is not None and float(ratio)<1:chains[int(tokens)]=choice
recipes={}
for tokens,(front,tail_on) in chains.items():
 for phase in (('b1','pool') if front else ())+ (('b2',) if tail_on else ()):
  if (tokens,phase) in stages:recipes[tokens,phase]=[stages[tokens,phase]]
print('Positive full-chain shapes',list(chains),'shortlisted phases',list(recipes))
if a.recipes:a.recipes.write_text('GB_H200_RECIPES = '+pprint.pformat(recipes,width=110,sort_dicts=False)+'\n')
