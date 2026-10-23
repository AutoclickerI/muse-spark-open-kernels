"""Summarize a DIAG4-style log: per shape, cuBLAS vs skinny vs best wgmma config."""
import re, sys
txt = open(sys.argv[1]).read()
for m in re.finditer(r"WG (\w+) m(\d+) cublas ([\d.na]+) skinny ([\d.na]+) \| (.*)", txt):
    layer, mm, cb, sk, rest = m.groups()
    cfgs = re.findall(r"([\d.]+):(\d+)", rest)
    best = cfgs[0] if cfgs else ("nan", "")
    ref = float(cb) if cb != "nan" else float("nan")
    print(f"{layer:5s} m{mm:>4s} cublas {cb:>6s} skinny {sk:>6s} wg {best[0]:>6s} ({best[1]})  "
          + " ".join(f"{t}:{c}" for t, c in cfgs[1:4]))
for line in txt.splitlines():
    if "WGBAD" in line or "PASSED" in line or "FAILED" in line or "failed" in line:
        print(line)
