"""Force the q_len=8 split-K attention choice through the full layer ABI."""

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
    cand = _load("verify_dropin", PDIR / "dev/cand_r3_v9_attention_dropin.py")
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    state = cand.prepare(weights)
    for batch in (16, 32):
        data = ref.generate_input(batch=batch, ctx=80000, layer="full", q_len=8,
                                  spread="agentic", page=1, dist="soft", seed=19000 + batch)
        state["attention_plan"][(batch, 8, False)] = ("port_verify", None)
        out = cand.kernel_fn(data, state)
        torch.cuda.synchronize()
        raw = ref.collect(data, out)
        ok, msg = ref.check_implementation(data, raw, weights)
        assert ok, (batch, msg)
        print(f"verify_full_b{batch}: PASS {msg}")


if __name__ == "__main__":
    main()
