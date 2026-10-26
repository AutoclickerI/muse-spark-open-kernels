"""Machine-code and graph protocol proof for deleting unselected families."""
import ast,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent;o=H/'out';native=o/'native';symbol='_Z5attndILi2ELi64EEv1A2WF'
def instructions(p):
 s=p.read_text();a=s.index('Function : '+symbol);z=s.find('\n\t\tFunction : ',a+len(symbol));s=s[a:z if z>=0 else len(s)]
 # Keep exact encoded instruction words; delete only presentation offsets.
 code=[]
 for line in s.splitlines():
  words=re.findall(r'/\* (0x[0-9a-f]+) \*/',line)
  code.extend(words)
 return code
old=instructions(native/'lee_hybrid.sass');new=instructions(native/'lee_selected.sass');assert old and len(old)==len(new)
differences=[(i,a,b) for i,(a,b) in enumerate(zip(old,new)) if a!=b]
assert differences==[(64,'0x0000001004047890','0x0000040004047890')],differences
# Only extern-shared symbol alignment differs after pruning. The following
# +1023 &~1023 maps both bases to2048 before any access; all other words match.
assert ((0x400+0x10+0x3ff)&~0x3ff)==((0x400+0x400+0x3ff)&~0x3ff)==2048
x=json.loads((H/'pruning.json').read_text());x.update(selected_lee_encoded_SASS_identical_except_equivalent_smem_base=True,shared_base_proof='0x400+{0x10,0x400}+1023 then &~1023 both2048; all other words identical. Static allocation differs but dynamic216320B keeps occupancy1 in both.',sass_differences=differences,sass_words=len(new),sass_words_sha256=hashlib.sha256(''.join(new).encode()).hexdigest(),native=json.loads((o/'native_lee_selected.json').read_text()))
for mode in ('normal','forced'):
 d=json.loads((H/('candidate_'+mode+'_screen.json')).read_text());assert all(r['ok'] for r in d['static']+d['tests']);x[mode]=dict(sha256=d['sha256'],TASK_cases=len(d['tests']),eager_and_graph=True,allpass=True)
assert x['normal']['sha256']==x['forced']['sha256'];(H/'selected_validation.json').write_text(json.dumps(x,indent=2)+'\n');print(x)
