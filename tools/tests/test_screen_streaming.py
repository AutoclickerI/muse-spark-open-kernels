import contextlib
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screen_all


class ScreenStreamingRegression(unittest.TestCase):
    def test_fast_pass_is_sent_before_slow_screen_finishes(self):
        fast_sent = threading.Event()
        sent = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slow, fast = root/'slow.py', root/'fast.py'
            def screen(file, gpu, logdir):
                if file == slow:
                    if not fast_sent.wait(timeout=2):
                        raise AssertionError('send blocked behind unfinished screen')
                return dict(file=file.name, problem='muse_decode_layer', ok=True, tests='1/1',
                            graphs='1/1', minutes=0, why='', log='unused')
            def popen(command, **kwargs):
                sent.append(command[3])
                if command[3] == 'fast.py':
                    fast_sent.set()
            with patch.object(screen_all, 'REPO', root), patch.object(screen_all, 'screen', side_effect=screen), \
                 patch.object(screen_all, 'unsent', return_value=([slow, fast], [])), \
                 patch.object(screen_all.subprocess, 'Popen', side_effect=popen), \
                 patch.object(sys, 'argv', ['screen_all.py', '--send', '--jobs', '2']), \
                 contextlib.redirect_stdout(io.StringIO()):
                screen_all.main()
        self.assertEqual(sent, ['fast.py', 'slow.py'])


if __name__ == '__main__':
    unittest.main()
