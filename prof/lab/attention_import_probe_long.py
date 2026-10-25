"""Measure how long an attention worker accepts import-time GPU work: run until the payload deadline in 10 s steps."""
import time

_IMPORT_START = time.monotonic()

import torch


def lab(deadline, weights, state):
    assert weights is None and state is None, "this probe measures a host without prepare()"
    a = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    lines = [f"IMPORT_START budget_s={deadline - _IMPORT_START:.1f}"]
    print(lines[0], flush=True)
    step = 0
    while time.monotonic() < deadline - 1.0:
        step += 1
        until = min(_IMPORT_START + step * 10.0, deadline - 1.0)
        while time.monotonic() < until:
            for _ in range(32):
                torch.mm(a, b, out=out)
            torch.cuda.synchronize()
        lines.append(f"IMPORT_STEP step={step} elapsed_s={time.monotonic() - _IMPORT_START:.3f}")
        print(lines[-1], flush=True)
    lines.append(f"IMPORT_DONE elapsed_s={time.monotonic() - _IMPORT_START:.3f}")
    return lines
