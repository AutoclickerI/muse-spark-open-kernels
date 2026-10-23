"""Portable glue test for Hopper-only dispatch branches in cand_r3_v9_autotune."""
import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "problems/muse_decode_layer/cand_r3_v9_autotune.py"
spec = importlib.util.spec_from_file_location("decode_autotune_glue", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Portable FA3 stand-in: preserve the tensor ABI and shape, while avoiding Hopper binaries.
fake = types.ModuleType("sgl_kernel.flash_attn")
fake.get_scheduler_metadata = lambda **kwargs: None

def fake_flash(q, **kwargs):
    cfg = kwargs["_cfg"]
    return q.to(torch.bfloat16).view(cfg["batch"], cfg["q_len"], mod.HQ, mod.HD)

# The candidate does not pass cfg through the installed FA3 ABI, so wrap the stand-in with
# the fixed synthetic dimensions used below.
fake.flash_attn_with_kvcache = lambda q, **kwargs: q.to(torch.bfloat16).view(1, 8, mod.HQ, mod.HD)
sys.modules["sgl_kernel.flash_attn"] = fake

device = torch.device("cuda")
q = torch.randn((8, mod.HQ, mod.HD), device=device, dtype=torch.float16).to(mod.FP8)
gate = torch.randn((8, mod.Q_SIZE), device=device, dtype=torch.float32)
data = {
    "cache_seqlens": torch.full((1,), 80008, device=device, dtype=torch.int32),
    "cu_seqlens_q": torch.arange(0, 9, device=device, dtype=torch.int32),
    "page_table": torch.arange(8, device=device, dtype=torch.int32).view(1, 8),
    "k_cache": torch.zeros((8, 1, mod.HKV, mod.HD), device=device, dtype=mod.FP8),
    "v_cache": torch.zeros((8, 1, mod.HKV, mod.HD), device=device, dtype=mod.FP8),
    "k_scale": torch.ones((), device=device), "v_scale": torch.ones((), device=device),
    "config": {"batch": 1, "q_len": 8, "page_size": 1, "window": 2048,
               "sm_scale": 0.125, "qk_scale_factor": 5.473, "kv_scale": 1.0},
}
out = mod._fa3_call(q, gate, data, (8, True))
assert out.shape == (8, mod.Q_SIZE) and out.dtype == torch.bfloat16

class FakeLT:
    def lt_run(self, w, x, out, work, algos, index):
        out.copy_(torch.mm(x, w.t()).to(out.dtype))

x = torch.randn((16, 1024), device=device, dtype=torch.bfloat16)
w = torch.randn((4096, 1024), device=device, dtype=torch.bfloat16)
state = {"lt_mod": FakeLT(), "lt_workspace": torch.empty((32 << 20,), device=device, dtype=torch.uint8),
         "lt_algos": {("o", 16): torch.zeros((1, 16), device="cpu", dtype=torch.uint8)}}
y = mod._materialize(mod._project_variant(x, w, "o", state, ("lt", 0)))
assert y.shape == (16, 4096) and y.dtype == torch.bfloat16

torch.cuda.synchronize()
print("PASS portable FA3 ABI/setting glue and cuBLASLt plan glue", flush=True)
