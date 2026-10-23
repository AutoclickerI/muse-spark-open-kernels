"""Print compact, receipt-grounded H200 comparisons for this problem."""
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
ledger = [json.loads(line) for line in (repo / "runs/ledger.jsonl").read_text().splitlines()]
print("candidate | job | mode | overall | D1 | D16 | D32 | V1 | V16 | V32")
for row in ledger:
    if row["problem"] != "muse_moe_w4a16":
        continue
    receipt = repo / Path(row["source"]).with_suffix(".json")
    result = json.loads(receipt.read_text()).get("result") or {}
    ratios = {r["case"]["bs"]: r["speedup"] for r in result.get("records", [])
              if r.get("event") == "benchmark" and "speedup" in r}
    nums = [result.get("speedup")] + [ratios.get(bs) for bs in (1, 16, 32, 8, 128, 256)]
    print(" | ".join([Path(row["file"]).stem, row["job"], row["mode"]] +
                     [f"{n:.3f}" if n is not None else "-" for n in nums]))
    if not result.get("passed"):
        print("  ERROR:", str(result.get("error"))[:600])
