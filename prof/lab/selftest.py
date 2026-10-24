"""Lab self-test: device facts and one BF16 GEMM timing, to prove the lab path end to end."""
import torch


def lab(deadline, weights):
    import time
    p = torch.cuda.get_device_properties(0)
    out = [f"LAB device={p.name} sm={p.major}{p.minor} sms={p.multi_processor_count} mem_gb={p.total_memory/1e9:.1f}",
           f"LAB weights={sorted(weights) if isinstance(weights, dict) else type(weights).__name__}"]
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        a @ b
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 100
    out.append(f"LAB gemm8192 us={us:.1f} tflops={2*8192**3/us/1e6:.0f} left_s={deadline-time.monotonic():.0f}")
    return out
