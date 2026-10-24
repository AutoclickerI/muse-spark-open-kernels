"""Local full-layer gate for the q_len=1 FP8-KV split decode choice."""

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
PDIR = ROOT / "problems/muse_decode_layer"
sys.path.insert(0, str(PDIR))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = _load("reference", PDIR / "reference.py")
    cand = _load("decode_fp8_split_dropin", PDIR / "dev/cand_r3_v9_attention_dropin.py")
    # The local sm_120 image cannot execute the Hopper-only scheduler probe
    # used only to populate full q1 metadata; the kernel gate itself is exact.
    ref._fa3_scheduler_metadata = lambda *args, **kwargs: None
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    state = cand.prepare(weights)
    for batch in (1, 16, 32):
        for layer in ("sliding", "full"):
            data = ref.generate_input(batch=batch, ctx=80000, layer=layer, q_len=1,
                                      spread="agentic", page=1, dist="normal",
                                      seed=19400 + batch + (0 if layer == "sliding" else 100))
            print(f"running decode1_{layer}_b{batch}", flush=True)
            state["attention_plan"][(batch, 1, layer == "sliding")] = ("port_decode_fp8_split", None)
            out = cand.kernel_fn(data, state)
            torch.cuda.synchronize()
            raw = ref.collect(data, out)
            ok, msg = ref.check_implementation(data, raw, weights)
            assert ok, (batch, layer, msg)
            print(f"decode1_{layer}_b{batch}: PASS {msg}")


if __name__ == "__main__":
    main()
