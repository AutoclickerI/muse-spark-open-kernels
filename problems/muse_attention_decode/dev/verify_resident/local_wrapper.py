"""Suppress import-only lab execution off Hopper for the local_eval wrapper; all timed code is unchanged."""
from pathlib import Path
H=Path(__file__).resolve().parent;s=(H/'out/board.py').read_text();at=s.index('import time as _lab_time\n')
s=s[:at]+"if torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9:\n"+'\n'.join('    '+l for l in s[at:].splitlines())+'\n'
(H/'out/board_local.py').write_text(s)
