"""Measure whether an attention worker accepts another 45 seconds of import-time GPU work."""
import time

_IMPORT_START = time.monotonic()

import torch


def lab(deadline, weights, state):
    assert weights is None and state is None, "this probe measures a host without prepare()"
    lines = [f"IMPORT_START monotonic={_IMPORT_START:.6f}"]
    print(lines[0], flush=True)
    a = torch.randn((1024, 1024), dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(20):
        torch.mm(a, b, out=out)
    end.record()
    end.synchronize()
    lines.append(f"CUBLAS_GEMM mean_ms={start.elapsed_time(end) / 20:.6f}")
    print(lines[-1], flush=True)
    stop = _IMPORT_START + 45.0
    if deadline < stop:
        raise RuntimeError("probe needs 45 seconds remaining; send it alone with --import-budget >= 60")
    for step in range(1, 10):
        until = min(_IMPORT_START + step * 5.0, stop)
        while time.monotonic() < until:
            # Small synchronized batches keep the GPU busy without queuing beyond the deadline.
            for _ in range(32):
                torch.mm(a, b, out=out)
            torch.cuda.synchronize()
        lines.append(f"IMPORT_STEP step={step} elapsed_s={time.monotonic() - _IMPORT_START:.3f}")
        print(lines[-1], flush=True)
    lines.append(f"IMPORT_DONE elapsed_s={time.monotonic() - _IMPORT_START:.3f}")
    return lines
