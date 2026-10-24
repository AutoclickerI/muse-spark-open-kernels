"""Build the decode-layer attention drop-in from the v9-derived portfolio.

The generated candidate keeps the layer portfolio's prepare-time attention tuner, but
publishes its shape key and removes stale prefix-fusion code that the new static checker
rightly rejects.  Attention choices are keyed by the twelve decode-layer shapes:
batch 1/16/32, q_len 1/8, and sliding/full.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PORTFOLIO = REPO / "problems/muse_decode_layer/cand_r3_v9_attention_portfolio.py"
OUTPUT = REPO / "problems/muse_decode_layer/dev/cand_r3_v9_attention_dropin.py"

SHAPES = tuple((batch, q_len, sliding) for batch in (1, 16, 32) for q_len in (1, 8) for sliding in (True, False))


def _assignment(source: str, name: str, renamed: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == name:
                lines = source.splitlines(keepends=True)
                chunk = source[sum(len(x) for x in lines[: node.lineno - 1]): sum(len(x) for x in lines[: node.end_lineno])]
                return re.sub(r"^" + re.escape(name) + r"\s*=", renamed + " =", chunk, count=1, flags=re.M)
    raise KeyError(name)


def _remove_prefix_fusion(source: str) -> str:
    # The fused prefix helper calls a deleted o_and_pool implementation. Keep
    # the regular measured path for both tuner labels so the selector remains
    # shape-stable without carrying an undefined dead branch.
    source = re.sub(
        r"\n\ndef _prefix_variant\(.*?\n\ndef _tune_prefix",
        '''

def _prefix_variant(attn, stream, state, data, kind, use_pdl):
    t = attn.shape[0]
    sync = torch.zeros((15 * t,), device=attn.device, dtype=torch.int32)
    si = sync[:3 * t].view(3, t)
    sf = sync[3 * t:].view(torch.float32).view(3, t, 4)
    a = project(attn, state["o"], "o", state)
    h, stream1 = boundary(a, stream, state["alpha1"], state["beta1"], state["pre_ffn_w"],
                          (sf[0], si[0]), use_pdl=use_pdl)
    pr = project(h, state["pool_router"], "pool_router", state, out_dtype=torch.float32)
    sync2 = torch.zeros((15 * t,), device=attn.device, dtype=torch.int32)
    si2 = sync2[:3 * t].view(3, t)
    sf2 = sync2[3 * t:].view(torch.float32).view(3, t, 4)
    return route_and_norm(pr, state, data, (sf2[1], si2[1])) + (stream1,)


def _tune_prefix''',
        source,
        flags=re.S,
    )
    source = source.replace(
        "    prefix = state.get(\"prefix_plan\", {}).get(tokens, (\"regular\", True))\n    if prefix[0] == \"regular\":\n        return _kernel_regular(data, state)\n    dev = data[\"stream\"].device\n",
        "    return _kernel_regular(data, state)\n    # The regular path is the measured v9 path for all twelve layer shapes.\n    dev = data[\"stream\"].device\n",
    )
    # Remove the now-unreachable non-regular body after the early return.
    source = re.sub(
        r"    return _kernel_regular\(data, state\)\n    # The regular path is the measured v9 path for all twelve layer shapes\.\n    dev = data\[\"stream\"\]\.device\n.*?\n\n\n# -------------------------------------------------------------------------------------------------\n# Attention-lane portfolio port",
        "    return _kernel_regular(data, state)\n\n\n# -------------------------------------------------------------------------------------------------\n# Attention-lane portfolio port",
        source,
        flags=re.S,
    )
    return source


def build(output: Path = OUTPUT) -> Path:
    source = _remove_prefix_fusion(PORTFOLIO.read_text())
    source = source.replace(
        "_measure_graph(lambda setting=setting: _fa3_call(q, gate, data, setting), flush, repeats=3)",
        "_measure_graph(lambda q=q, gate=gate, data=data, setting=setting: _fa3_call(q, gate, data, setting), flush, repeats=3)",
    )
    source = source.replace(
        "fn = lambda choice=choice: _attn_port_variant(qkv, data, state, choice)",
        "fn = lambda qkv=qkv, data=data, choice=choice: _attn_port_variant(qkv, data, state, choice)",
    )
    source = source.replace(
        'LT_CUDA_SRC = r"""\n#include <torch/extension.h>',
        'LT_CUDA_SRC = r"""\n#include <ATen/cuda/CUDAContext.h>\n#include <torch/extension.h>',
    )
    source = source.replace("_ATTN_PORT_X_EXT = {}\n", "_ATTN_PORT_EXT = None\n_ATTN_PORT_X_EXT = {}\n", 1)

    # The portfolio's Hopper dispatch refers to the standalone XQA families but
    # the old source omitted their literal CUDA assignments. Embed the screened
    # WGMMA source strings so local_eval compiles every branch for sm90a.
    wgmma = (REPO / "problems/muse_attention_decode/cand_attn_verify_wgmma.py").read_text()
    source += "\n\n# ---- embedded screened XQA/WGMMA source strings ----\n"
    for old, new in (("X_CUDA", "ATTN_X_CUDA"), ("X_CPP", "ATTN_X_CPP"),
                     ("G_CUDA", "ATTN_G_CUDA"), ("P8_CUDA", "ATTN_P8_CUDA")):
        source += _assignment(wgmma, old, new) + "\n"

    source += f"""

# ---- attention_dropin public shape contract ----
ATTENTION_DROPIN_SHAPES = {SHAPES!r}

def _cos_sin(theta, d, device):
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    freq = torch.arange(1 << 20, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat((freq.cos(), freq.sin()), dim=-1)

def attention_dropin_shape(data):
    cfg = data["config"]
    return (cfg["batch"], cfg["q_len"], bool(cfg["window"]))

def attention_dropin_choice(state, data):
    key = attention_dropin_shape(data)
    return state.get("attention_plan", {{}}).get(key, ("layer", None))
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    return output


if __name__ == "__main__":
    print(build())
