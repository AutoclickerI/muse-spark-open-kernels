import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))


class LocalTierRegression(unittest.TestCase):
    def test_static_preflight_does_not_import_or_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.py'
            p.write_text('# okc: problem=muse_decode_layer\nraise RuntimeError("executed candidate")\n'
                         'def prepare(weights):return {"present": 1}\n'
                         'def kernel_fn(data,state):return state["absent"] + unknown_name\n')
            out = Path(tmp) / 'report.json'
            run = subprocess.run([sys.executable, str(TOOLS/'local_eval.py'), str(p), '--tier', 'static', '--json', str(out)],
                                 capture_output=True, text=True)
            self.assertTrue(out.exists(), run.stdout+run.stderr)
            result = json.loads(out.read_text())
            self.assertEqual(run.returncode, 1)
            self.assertFalse(any(c['ok'] for c in result['static'] if c['name'] in ('undefined_names', 'state_keys')))
            self.assertNotIn('executed candidate', run.stderr)
            self.assertNotIn('prepare_s', result)

    def test_changed_units_only_compile_changed_source(self):
        import local_eval
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp)/'old.py', Path(tmp)/'new.py'
            base = '# okc: problem=muse_decode_layer\nCUDA_A="old"\nCUDA_B="same"\ndef prepare(w):return {}\ndef kernel_fn(d,s):return d\n'
            old.write_text(base)
            new.write_text(base.replace('CUDA_A="old"', 'CUDA_A="new"'))
            with patch.object(local_eval, 'hopper_compile', return_value=[]) as compile_check, patch.object(sys, 'argv', [
                'local_eval.py', str(new), '--tier', 'changed', '--against', str(old)]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as done:
                    local_eval.main()
            self.assertEqual(done.exception.code, 0)
            self.assertEqual(compile_check.call_args.kwargs['units'], {'CUDA_A'})

    def test_wrapper_smoke_each_transport(self):
        import local_eval
        for transport in ('prepare', 'import', 'lab-only'):
            with self.subTest(transport=transport):
                result = local_eval.wrapper_smoke(transport)
                self.assertTrue(result['ok'], result)
                self.assertEqual(result['packets'], 1)

    def test_packaged_state_and_signature_errors_are_seen(self):
        import base64
        import hashlib
        import lzma
        from local_preflight import preflight
        body = 'def prepare(w):return {"a":1}\ndef kernel_fn(d,s):return s["b"]\ndef fn(x):return x\nfn(bad=1)\n'
        encoded = base64.b85encode(lzma.compress(body.encode())).decode()
        src = '# okc: problem=muse_decode_layer\nimport base64,lzma,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
        src += 'assert hashlib.sha256(_source.encode()).hexdigest()=='+repr(hashlib.sha256(body.encode()).hexdigest())+'\nexec(compile(_source,"<packed>","exec"),globals())\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'packed.py'
            path.write_text(src)
            checks = preflight(path)
            self.assertTrue(any(c['name'] == 'state_keys' and not c['ok'] for c in checks))
            self.assertTrue(any(c['name'] == 'call_signatures' and not c['ok'] for c in checks))


if __name__ == '__main__':
    unittest.main()
