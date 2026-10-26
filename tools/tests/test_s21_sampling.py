"""Prospective timing-plan admission and all-round ABBA aggregation; no GPU needed."""
import copy
import importlib.util
from pathlib import Path
import statistics
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from promotion import _protocol_arm_valid


class SamplingProtocol(unittest.TestCase):
    def test_declared_counts_and_no_optional_stopping(self):
        plan=dict(name='palindromic-cohort-v1',rounds=16,seed_count=64,
                  aggregation='arithmetic-mean-all-rounds',selective_retiming=False)
        m=dict(measurement_protocol=plan,arms={'INC':dict(rows=[])})
        labels=[('d' if q==1 else 'v')+str(b)+s for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for s in ('S','F')]
        r=dict(measurement_protocol=copy.deepcopy(plan),timing_round_counts={l:{'INC':[16]*64} for l in labels})
        self.assertTrue(_protocol_arm_valid(r,m,'INC',64))
        for mutate in (lambda d:d['timing_round_counts']['v16S']['INC'].__setitem__(63,15),
                       lambda d:d['measurement_protocol'].__setitem__('selective_retiming',True),
                       lambda d:d['measurement_protocol'].__setitem__('seed_count',63)):
            bad=copy.deepcopy(r);mutate(bad);self.assertFalse(_protocol_arm_valid(bad,m,'INC',64))
        self.assertTrue(_protocol_arm_valid({},dict(arms={}), 'INC',16))

    def timing_namespace(self):
        import math,time
        ns=dict(statistics=statistics,math=math,time=time)
        exec((ROOT/'problems/muse_decode_layer/dev/s21/timing.py.inc').read_text(),ns)
        return ns

    def test_palindromes_cancel_linear_slot_drift(self):
        ns=self.timing_namespace()
        for names in (['A','INC'],['A','A2','INC','no_C3','no_GATE']):
            for seed in range(64):
                orders=list(ns['s21_orders'](names,16,seed));positions={n:[] for n in names}
                for block in range(8):
                    self.assertEqual(orders[2*block],orders[2*block+1][::-1])
                for r,order in enumerate(orders):
                    for slot,n in enumerate(order):positions[n].append(r*len(names)+slot)
                # A linear additive slot bias has the same mean for every arm.
                self.assertEqual(len({statistics.mean(v) for v in positions.values()}),1)

    def test_spike_is_kept_in_mean_and_trace(self):
        ns=self.timing_namespace();clock=[0.];which=[0];arm_round={'A':0,'A2':0,'INC':0}
        class Event:
            def __init__(self,**kw):pass
            def record(self):self.t=clock[0]
            def elapsed_time(self,end):return (end.t-self.t)/1000
        def replay(name):
            arm_round[name]+=1;clock[0]+=1000. if name=='INC' and arm_round[name]==7 else 100.
        cuda=types.SimpleNamespace(Event=Event,synchronize=lambda:None)
        ns.update(torch=types.SimpleNamespace(cuda=cuda),lab_cold_l2=lambda:None)
        ns['s21_telemetry']=lambda:dict(sm_MHz=1980)
        ns['S21_TRACE_CONTEXT'].update(label='v16S',seed=123,seed_index=0)
        g={n:(types.SimpleNamespace(replay=lambda n=n:replay(n)),None) for n in arm_round}
        out=ns['ab_time'](g,16)
        self.assertEqual(out['INC'][6],1000.)
        self.assertEqual(statistics.mean(out['INC']),156.25)
        self.assertEqual(ns['S21_ROUND_COUNTS']['v16S']['INC'],[16])
        self.assertTrue(ns['S21_TOP_TRACES'][0]['flagged'])
        self.assertEqual(len(ns['S21_TOP_TRACES'][0]['round_us']['INC']),16)

    def test_sample_codec_keeps_every_quantized_value(self):
        path=ROOT/'problems/muse_decode_layer/dev/s21/evidence.py'
        ns={};exec(path.read_text(),ns)
        names=['A','A2','INC','leaveout']
        loads={'decode 1':dict(n=64,samples=[dict(A=1.,A2=1+i/1e6,INC=1.3 if i==63 else .999,leaveout=1+i/1e6) for i in range(64)]),
               'verify 16':dict(n=64,samples=[dict(A=1.,A2=1-i/1e6,INC=1.,leaveout=1-i/1e6) for i in range(64)])}
        packed=ns['s21_pack_samples'](loads,names,1000000)
        self.assertLess(len(packed['sample_vectors']),len(loads)*len(names))
        raw=dict(sample_arms=names,sample_loads=list(loads),sample_scale=1000000,loads={k:dict(n=v['n']) for k,v in loads.items()},**packed)
        decoded=ns['s21_decode_samples'](raw)
        for label,cell in loads.items():
            for before,after in zip(cell['samples'],decoded['loads'][label]['samples']):
                for n in names:self.assertAlmostEqual(after[n],round(before[n]*1e6)/1e6)

    def test_receipt_columns_preserve_every_scalar_and_round(self):
        ns={};exec((ROOT/'problems/muse_decode_layer/dev/s21/evidence.py').read_text(),ns)
        raw=dict(complete=True,failures=[],integer=[0,-1,2**40],decimal=[0.,-1.000001,1234567.000001],
                 other=[None,True,False,'text'],round_us={'A':[32.,1057.792067,.123456]},
                 host_enqueue_us={'A':[1.234567,3.,100000.001]},nested={'round_us':{'INC':[.000001,650.272012]}})
        self.assertEqual(ns['s21_unpack_evidence'](ns['s21_pack_evidence'](raw)),raw)

    def test_receipt_columns_keep_post_measurement_device_fault(self):
        ns={};exec((ROOT/'problems/muse_decode_layer/dev/s21/evidence.py').read_text(),ns)
        from lab_packet import encode_packet,paired_evidence
        original=dict(complete=True,failures=[dict(scope='arm',arms=['optional'],reason='TASK')],round_us={'A':[100.]})
        packet=dict(kind='S21',paired=ns['s21_pack_evidence'](original),transport_errors=['sticky device fault'])
        recovered=ns['s21_unpack_evidence'](paired_evidence(encode_packet('s21','fixture',packet),'S21'))
        self.assertFalse(recovered['complete'])
        self.assertEqual(recovered['failures'][0],original['failures'][0])
        self.assertEqual(recovered['failures'][-1]['scope'],'device')

    def test_actual_gate_rejects_missing_primary_rounds_and_isolates_optional(self):
        spec=importlib.util.spec_from_file_location('s21_archive_fixture',ROOT/'tools/tests/test_promotion.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        obj=fixture.PromotionRegression();obj.setUp();raw=obj.scoped();manifest=copy.deepcopy(obj.manifest)
        plan=dict(name='palindromic-cohort-v1',rounds=16,seed_count=16,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)
        manifest['measurement_protocol']=plan;raw['measurement_protocol']=copy.deepcopy(plan)
        labels=[('d' if q==1 else 'v')+str(b)+s for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for s in ('S','F')]
        raw['timing_round_counts']={l:{n:[16]*16 for n in raw['arm_status']} for l in labels}
        from promotion import evaluate
        receipt=dict(id='protocol-integration',result=dict(passed=True))
        self.assertTrue(evaluate(receipt,raw,manifest)['freeze_gate'])
        bad=copy.deepcopy(raw);bad['timing_round_counts']['v16S']['INC'][-1]=8
        self.assertFalse(evaluate(receipt,bad,manifest)['measured'])
        bad=copy.deepcopy(raw);bad['timing_round_counts']['v16S']['no_BLOCK'][-1]=8
        result=evaluate(receipt,bad,manifest)
        self.assertTrue(result['freeze_gate']);self.assertFalse(result['pieces']['no_BLOCK']['measured'])


if __name__=='__main__':unittest.main()
