"""Build a synthetic-weight verify-streaming lab for the MoE host."""

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
source = (HERE.parent / "verify_streaming_lab.py").read_text()
driver = (HERE / "synthetic_moe_lab.inc").read_text()
body = "# okc: problem=muse_moe_w4a16\n"
body += "VERIFY_SOURCE = " + repr(source) + "\n"
body += driver
out = ROOT / "prof/lab/verify_streaming_synthetic_moe.py"
out.write_text(body)
print(out)
print("sha256", hashlib.sha256(body.encode()).hexdigest())
