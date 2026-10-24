"""Force the FP8-probability/PV verify choice through the layer ABI."""

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
    cand = _load("fp8_verify_dropin", PDIR / "dev/cand_r3_v9_attention_dropin.py")
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    state = cand.prepare(weights)
    for waves in (1, 2):
        for batch in (16, 32):
            data = ref.generate_input(batch=batch, ctx=80000, layer="full", q_len=8,
                                      spread="agentic", page=1, dist="soft", seed=19300 + batch + waves)
            state["attention_plan"][(batch, 8, False)] = ("xqa_fp8_pv", (3, waves))
            out = cand.kernel_fn(data, state)
            torch.cuda.synchronize()
            raw = ref.collect(data, out)
            ok, msg = ref.check_implementation(data, raw, weights)
            assert ok, (waves, batch, msg)
            print(f"fp8_pv_w{waves}_full_b{batch}: PASS {msg}")


if __name__ == "__main__":
    main()
