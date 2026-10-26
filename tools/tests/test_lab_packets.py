import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h200_lab as lab


class LabPacketRegression(unittest.TestCase):
    def base(self, tmp):
        path = Path(tmp) / 'base.py'
        path.write_text('# okc: problem=muse_bf16_proj\n'
                        'def prepare(weights):\n    print("host tuner" * 3000)\n    return weights\n')
        return path

    def execute(self, src):
        out = io.StringIO()
        cuda = types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
        ns = {'torch': types.SimpleNamespace(cuda=cuda)}
        with contextlib.redirect_stdout(out):
            exec(src, ns)
            ns['prepare'](None)
        return out.getvalue()

    def test_max_packet_survives_wrapper_and_receipt_tail(self):
        # Incompressible declared maximum; tune padding to fill the wire allocation.
        import lab_packet
        rng = random.Random(19)
        packet = dict(identities={'candidate': 'a'*64, 'incumbent': 'b'*64},
                      seeds=list(range(16)), paired=[{'old': 1.1, 'new': 1.0}]*16,
                      checks={'TASK': True, 'nearby': True}, complete=True, padding='')
        data = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=20000))
        low, high = 0, len(data)
        while low < high:
            mid = (low + high + 1)//2
            packet['padding'] = data[:mid]
            if len(lab_packet.encode_packet('maximum', 'abc', packet).encode())+1 <= lab.LIMIT-256:
                low = mid
            else:
                high = mid-1
        packet['padding'] = data[:low]
        payload = 'LAB_RESOURCES='+repr(dict(compile_s=1, run_s=1, output_bytes=lab.LIMIT-256, context_poison='none'))+'\n'
        payload += 'def lab(deadline, weights):\n    return '+repr({'packet': packet, 'logs': ['noisy log '*4000]})+'\n'
        with tempfile.TemporaryDirectory() as tmp:
            src, _ = lab.build([('maximum', 'abc', payload)], 'muse_bf16_proj', base_file=self.base(tmp))
            output = self.execute(src)
        tail = (output + 'P'*6000)[-16000:]
        packets = lab_packet.read_packets(tail)
        self.assertEqual(packets[0]['packet'], packet)
        self.assertLessEqual(len(output[output.index('LAB_BEGIN'):].encode()), lab.LIMIT)
        damaged = tail.replace('LAB_PACKET maximum abc ', 'LAB_PACKET maximum abc 0', 1)
        with self.assertRaises(ValueError):
            lab_packet.read_packets(damaged)

    def test_more_than_two_small_payloads_and_resource_splits(self):
        def payload(i, **overrides):
            resources = dict(compile_s=1, run_s=2, output_bytes=700, context_poison='none')
            resources.update(overrides)
            return str(i), str(i), 'LAB_RESOURCES='+repr(resources)+'\ndef lab(d,w):return ["ok"]\n'
        entries = [payload(i) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            src, _ = lab.build(entries, 'muse_bf16_proj', base_file=self.base(tmp))
            self.assertEqual(self.execute(src).count('LAB_END'), 5)
        self.assertEqual(len(lab.plan_batches(entries, budget_s=155)), 1)
        self.assertEqual([len(b) for b in lab.plan_batches([payload(0, run_s=90), payload(1, run_s=90)], budget_s=155)], [1, 1])
        self.assertEqual([len(b) for b in lab.plan_batches([payload(0, output_bytes=5000), payload(1, output_bytes=5000)], budget_s=155)], [1, 1])
        self.assertEqual([len(b) for b in lab.plan_batches([payload(0, context_poison='device'), payload(1)], budget_s=155)], [1, 1])

    def test_lab_only_fits_maximum_packet_and_two_peers(self):
        entries=[]
        for name, size in [('maximum', lab.LIMIT-256), ('optional', 768), ('peer', 512)]:
            resources=dict(compile_s=0,run_s=1,output_bytes=size,context_poison='none')
            entries.append((name, 'abc', 'LAB_RESOURCES='+repr(resources)+'\ndef lab(d,w):return ["ok"]'))
        source, _ = lab.build(entries, 'muse_bf16_proj', lab_only=True)
        self.assertIn("('peer', 'abc'", source)
        self.assertEqual(len(lab.plan_batches(entries, budget_s=30, output_budget=lab.LAB_ONLY_LIMIT)), 1)
        self.assertEqual(len(lab.plan_batches(entries, budget_s=30)), 2)

    def test_reemitted_packets_validate_each_copy_and_preserve_final_batch_order(self):
        import lab_packet
        packets=[lab_packet.encode_packet(name,'sha',dict(complete=True,value=name)) for name in ('maximum','optional','peer')]
        # The receipt may cut the early maximum packet while retaining its peers and the late complete batch.
        records=lab_packet.read_packets('\n'.join(packets[1:]+packets))
        self.assertEqual([r['name'] for r in records],['maximum','optional','peer'])
        corrupted=packets[0].replace(' sha ',' sha 0',1)
        with self.assertRaises(ValueError):lab_packet.read_packets(corrupted+'\n'+packets[0])
        conflicting=lab_packet.encode_packet('maximum','sha',dict(complete=True,value='different'))
        with self.assertRaisesRegex(ValueError,'conflicting'):lab_packet.read_packets(packets[0]+'\n'+conflicting)

    def test_post_measurement_device_fault_invalidates_packet_evidence(self):
        import lab_packet
        raw = dict(complete=True, failures=[], arm_status={'A': {'complete': True}})
        packet = dict(kind='S18', identities={}, seeds=[1,2], paired=raw, checks={}, complete=False,
                      transport_errors=['LAB_ERROR CUDA illegal memory access during synchronize'])
        wire = lab_packet.encode_packet('fault', 'abc', packet)
        evidence = lab_packet.paired_evidence(wire, 'S18')
        self.assertFalse(evidence['complete'])
        self.assertTrue(evidence['failures'])
        self.assertEqual(evidence['failures'][0]['scope'], 'device')

    def test_cli_splits_output_and_preserves_send_flags(self):
        entries = [(str(i), str(i), 'LAB_RESOURCES='+repr(dict(compile_s=1, run_s=1, output_bytes=7000,
                    context_poison='none'))+'\ndef lab(d,w):return ["ok"]') for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'tools').mkdir()
            (root/'tools/lab_packet.py').write_bytes((lab.REPO/'tools/lab_packet.py').read_bytes())
            (root/'problems/muse_prefill_layer').mkdir(parents=True)
            with patch.object(lab, 'REPO', root), patch.object(lab, 'embed', side_effect=entries), \
                 patch.object(lab, 'quota', return_value={}), patch.object(lab, 'absorb_waiting',
                 side_effect=lambda h, e, p, n, **kw: (e,p,n)), \
                 patch.object(lab.subprocess, 'run') as send, patch.object(sys, 'argv', [
                    'h200_lab.py', 'send', '0.py', '1.py', '--host', 'muse_prefill_layer', '--lab-only',
                    '--import-budget', '700', '--lane', 'P4', '--decision', 'promote pair', '--note', 'batch',
                    '--not-before', '2026-10-27T00:00:00Z', '--priority', '-1', '--check']):
                lab.main()
            self.assertEqual(send.call_count, 2)
            for call in send.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index('--lane')+1], 'P4')
                self.assertEqual(command[command.index('--decision')+1], 'promote pair')
                self.assertIn('--check', command)
                self.assertIn('--not-before', command)
                # Explicit PLAN lane wins over a legacy numeric override.
                self.assertNotIn('--priority', command)

    def test_legacy_lab_send_after_P1_row_is_removed(self):
        import okc_run
        entry=('legacy','abc','def lab(d,w):return ["ok"]')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'tools').mkdir();(root/'tools/lab_packet.py').write_bytes((lab.REPO/'tools/lab_packet.py').read_bytes())
            (root/'problems/muse_bf16_proj').mkdir(parents=True)
            plan=root/'PLAN.md';plan.write_text('## Ranked work\n| 1 | P4 |\n')
            with patch.object(lab,'REPO',root),patch.object(okc_run,'PLAN',plan), \
                 patch.object(lab,'embed',return_value=entry),patch.object(lab,'quota',return_value={}), \
                 patch.object(lab,'absorb_waiting',side_effect=lambda h,e,p,n,**kw:(e,p,n)), \
                 patch.object(lab.subprocess,'run') as send,patch.object(sys,'argv',[
                    'h200_lab.py','send','legacy.py','--lab-only','--note','legacy decision']):
                lab.main()
            command=send.call_args.args[0]
            self.assertNotIn('--lane',command)
            self.assertNotIn('--check',command)  # non-decode lab-only sends spend spare scored quota
            self.assertEqual(command[command.index('--decision')+1],'legacy decision')

    def test_import_lab_only_avoids_base_prepare(self):
        with patch.object(lab, 'best_source', side_effect=AssertionError('base must not be loaded')):
            src, _ = lab.build([('own', 'abc', 'def lab(d,w):return ["self prepared"]')],
                               'muse_prefill_layer', lab_only=True)
        out = io.StringIO()
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))}):
            ns = {}
            with contextlib.redirect_stdout(out):
                exec(src, ns)
                with self.assertRaisesRegex(RuntimeError, 'lab-only host'):
                    ns['prepare'](None)
        self.assertIn('LAB_PACKET', out.getvalue())


if __name__ == '__main__':
    unittest.main()
