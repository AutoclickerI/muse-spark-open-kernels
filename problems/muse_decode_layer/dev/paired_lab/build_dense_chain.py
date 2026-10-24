"""Build the standalone best7 dense-chain paired lab payload."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "problems/muse_decode_layer/cand_r3_best7_reset.py"
DRIVER = Path(__file__).with_name("dense_chain_driver.py.inc")
OUT = ROOT / "prof/lab/dense_chain_best7_compare.py"

src = SOURCE.read_text()
body = "# okc: problem=muse_prefill_layer\n"
body += "BEST7_SOURCE = " + repr(src) + "\n"
body += DRIVER.read_text()
OUT.write_text(body)
print(OUT)
print("sha256", hashlib.sha256(body.encode()).hexdigest())
