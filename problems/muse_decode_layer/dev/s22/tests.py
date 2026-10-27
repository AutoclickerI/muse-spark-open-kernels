"""Selection cannot bypass fresh confirmation, and confirmation seeds cannot be reused."""
import copy,json,unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import compose,follow
H=compose.H
class SelectionTests(unittest.TestCase):
 def config(self):return dict(phase=0,host='muse_bf16_proj',dropped_cells=[],disabled_hybrids=[])
 def result(self):return dict(measured=True,lab_gate=True,cells={n:dict(measured=True,CI95=[1.01,1.02]) for n in compose.CELLS},hybrids={n:dict(measured=True,CI95=[1.01,1.02]) for n in compose.HYBRIDS})
 def test_resolved_cell_loss_needs_new_confirmation_even_if_primary_passed(self):
  r=self.result();r['cells']['v16S']['CI95']=[.98,.99];d=follow.decide(r,self.config());self.assertEqual(d['action'],'confirm');self.assertEqual(d['config']['dropped_cells'],['v16S']);self.assertEqual(d['config']['phase'],1)
 def test_hybrid_loss_keeps_beneficial_original_port(self):
  r=self.result();r['hybrids']['d32S']['CI95']=[.98,.99];d=follow.decide(r,self.config());self.assertEqual(d['config']['disabled_hybrids'],['d32S']);self.assertEqual(d['config']['dropped_cells'],[])
 def test_uncertain_cell_is_not_a_resolved_loss(self):
  r=self.result();r['cells']['v1F']['CI95']=[.99,1.01];self.assertEqual(follow.decide(r,self.config())['action'],'score')
 def test_failed_gate_has_no_automatic_repeat_or_score(self):
  r=self.result();r['lab_gate']=False;self.assertEqual(follow.decide(r,self.config())['action'],'stop')
 def test_incomplete_cell_evidence_cannot_score(self):
  r=self.result();r['cells']['v32S']['measured']=False
  with self.assertRaises(RuntimeError):follow.decide(r,self.config())
 def test_next_phase_seed_freshness(self):
  m=json.loads((H/'manifest.json').read_text());seen=set(m['selection_seeds']+m['confirmation_seeds']);next_seeds={303400003+104729*i for i in range(64)};self.assertFalse(seen&next_seeds)
 def test_composed_drop_does_not_change_other_cell_policies(self):
  c=compose.load(H.parent/'s19/contracts.py','s22_test_src');a=c.Src(compose.build()[0]);b=c.Src(compose.build({'v16S','v1F'},{'d32S'})[0]);p=a.literal('_PUBLIC_SELECTION');p.pop(compose.CELLS['v16S']);p.pop(compose.CELLS['v1F']);self.assertEqual(p,b.literal('_PUBLIC_SELECTION'));hy=a.literal('_HY_SELECTION');hy.pop(compose.CELLS['d32S']);self.assertEqual(hy,b.literal('_HY_SELECTION'))
 def test_scoring_requires_the_exact_confirmed_source(self):
  m=json.loads((H/'manifest.json').read_text())
  with patch.object(follow.runner,'run') as send:
   with self.assertRaises(AssertionError):follow.score_current(m,dict(lab_gate=True,source_sha256='different-source'))
   send.assert_not_called()
 def test_worker_waits_for_fresh_phase_before_scoring(self):
  with tempfile.TemporaryDirectory(dir=H/'generated') as root:
   temp=Path(root);(temp/'generated').mkdir();cfg=self.config()
   def write(name,value):(temp/name).write_text(json.dumps(value))
   def manifest(phase):return dict(phase=phase,host=cfg['host'],dropped_cells=[] if not phase else ['v16S'],disabled_hybrids=[],wrapper_sha256=f'wrapper{phase}',sha256=f'source{phase}')
   write('manifest.json',manifest(0));write('config.json',cfg);write('queue.json',dict(wrapper_sha256='wrapper0'))
   r0=self.result();r0.update(job='job0',source_sha256='source0',score=dict(R=1.01,CI95=[1.005,1.02]));r0['cells']['v16S']['CI95']=[.98,.99]
   r1=self.result();r1['cells'].pop('v16S');r1.update(job='job1',source_sha256='source1',score=dict(R=1.02,CI95=[1.015,1.03]))
   waits=[];scores=[]
   def wait(q):waits.append(q['wrapper_sha256']);return temp/'receipt.json'
   def run(cmd,*args,**kwargs):
    if str(cmd[1]).endswith('build_lab.py'):write('manifest.json',manifest(1))
   def send():
    q=dict(wrapper_sha256='wrapper1');write('queue.json',q);return q
   def score(m,r):scores.append(m['phase']);self.assertEqual(waits,['wrapper0','wrapper1']);return dict(pid=999)
   with patch.object(follow,'H',temp),patch.object(follow.signal,'signal'),patch.object(follow.runner,'wait_receipt',side_effect=wait),patch.object(follow.report,'analyze',side_effect=[r0,r1]),patch.object(follow.runner,'run',side_effect=run),patch.object(follow.runner,'validate_current'),patch.object(follow.runner,'send_current',side_effect=send),patch.object(follow,'score_current',side_effect=score),patch.object(follow,'freeze',return_value=dict(scored_speedup=1.9)),patch.object(follow.finish,'render'),patch.object(follow.finish,'commit'):
    follow.work('wrapper0')
   self.assertEqual(scores,[1]);self.assertEqual(json.loads((temp/'state.json').read_text())['status'],'frozen')
class PlanTests(unittest.TestCase):
 def test_plan_refresh_never_asserts_on_orchestrator_prose(self):
  for text in ('# Plan\n\nNo S22 bullet and no ranked table.\n','| Rank | Id |\n|---|---|\n| 3 | P5 | G1 | x | y | z | w |\n'):
   self.assertEqual(follow.finish.update_plan(text),text)
 def test_plan_refresh_touches_only_s22_bullet_and_p4_work(self):
  text='## Status\n- Board line.\n- S22 `old` (stale): text\n  continued.\n- Rule line.\n\n| Rank | Id | Serves | Work | Done | Who | State |\n|---|---|---|---|---|---|---|\n| 2 | P4 | G1 | old work | done | lane | active |\n'
  out=follow.finish.update_plan(text);lines=out.splitlines()
  self.assertIn('- Board line.',lines);self.assertIn('- Rule line.',lines);self.assertNotIn('  continued.',lines);self.assertNotIn('old work',out)
  row=next(x for x in lines if '| P4 |' in x).split('|');self.assertEqual([c.strip() for c in row[5:8]],['done','lane','active'])
if __name__=='__main__':unittest.main()
