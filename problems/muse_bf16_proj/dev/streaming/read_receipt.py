"""Build a decode-lane handoff from a landed receipt, without contacting the judge."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("receipt", help="job ID or local receipt path")
args = parser.parse_args()
receipt = Path(args.receipt)
if not receipt.exists():
    hits = list((REPO / "runs").glob(f"*/*_{args.receipt}.json"))
    if len(hits) != 1:
        raise SystemExit(f"expected one landed receipt, got {len(hits)}; no judge request made")
    receipt = hits[0]
data = json.loads(receipt.read_text())
tail = data["result"].get("log_tail", "")
section = re.search(r"^LAB_BEGIN streaming_gemv_lab ([0-9a-f]+)\n(.*?)(?=^LAB_END)", tail, re.M | re.S)
if section is None:
    raise SystemExit("streaming_gemv_lab section missing; inspect receipt error")
payload = HERE.parent / "streaming_gemv_lab.py"
sha = hashlib.sha256(payload.read_bytes()).hexdigest()
if not sha.startswith(section[1]):
    raise SystemExit("receipt payload hash differs from current source; use its archived payload")
if "SP_DONE" not in section[2] or "LAB_ERROR" in section[2]:
    raise SystemExit("lab did not complete cleanly; no handoff emitted")
rows = []
for line in section[2].splitlines():
    if not line.startswith("SP_PICK "):
        continue
    values = dict(re.findall(r"(\w+)=([^ ]+)", line))
    for key in ("m", "cfg", "split", "cta", "pass35", "reject"):
        values[key] = int(values[key])
    for key in ("cublas_us", "us", "ratio", "raw_TBs", "adj_TBs"):
        values[key] = float(values[key])
    values["role"] = line.split()[1]
    values["faster_than_cublas"] = values["ratio"] < 1.0
    rows.append(values)
out = dict(receipt=str(receipt.resolve().relative_to(REPO)), payload_sha256=sha,
           result_status=data.get("status"), shapes=rows,
           interpretation="Paired standalone medians, not integrated layer gains; small differences may be noise.",
           adapter="Construct StreamingProjection in prepare; return op(x).unsqueeze(0), 1. Keep pool/router hi+lo rows separate.")
dest = HERE / "handoff.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print(dest)
for row in rows:
    print(f"{row['role']:15s} M={row['m']:3d} cfg={row['cfg']:2d} split={row['split']} "
          f"{row['us']:.3f}us / cuBLAS {row['cublas_us']:.3f}us ratio={row['ratio']:.4f} "
          f"adjusted={row['adj_TBs']:.3f} TB/s")
