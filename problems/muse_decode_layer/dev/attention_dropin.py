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


def with_sliding_attention(source: str) -> str:
    """Add the gated one-launch choice; promotion still comes from a measured settings row."""
    marker = "def _attn_port_variant(qkv, data, state, choice):\n"
    if source.count(marker) != 1:
        raise ValueError("expected one attention portfolio dispatcher")
    branch = (
        marker
        + "    if choice[0] == 'singlelaunch_sliding':\n"
        + "        return attention_sliding(qkv, data)\n"
    )
    source = source.replace(marker, branch, 1)
    module = REPO / "problems/muse_decode_layer/dev/sliding_attention.py"
    return source + "\n\n# One-launch sliding attention; settings promotion awaits H200 evidence.\n" + module.read_text()



def with_verify8_flashdecode(source: str) -> str:
    """Add the full-q8 measurement choice; select only after an H200 win."""
    marker = "def _attn_port_variant(qkv, data, state, choice):\n"
    if source.count(marker) != 1:
        raise ValueError("expected one attention portfolio dispatcher")
    branch = (
        marker
        + "    if choice[0] == 'flashdecode_verify8':\n"
        + "        return attention_verify8(qkv, data, choice[1])\n"
    )
    source = source.replace(marker, branch, 1)
    module = REPO / "problems/muse_decode_layer/dev/verify8_flashdecode.py"
    return source + "\n\n# Full-q8 split attention; select only after H200 admission.\n" + module.read_text()


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
    # q_len=8 keeps all eight rows in each CTA, so the Hopper combine is the
    # limiter at the two-wave target. Use one resident wave for q8 and floor
    # the per-request split count (B16/B32 => 8/4 on a 132-SM H200); q1 keeps
    # the original two-wave ceiling.
    source = source.replace(
        "    target = 2 * sms\n    ns = -(-target // (batch * kvg))",
        "    target = sms if q_len == 8 else 2 * sms\n"
        "    ns = (target // (batch * kvg)) if q_len == 8 else -(-target // (batch * kvg))",
    )
    # The decode candidate keeps two resident waves only for B=1, where one
    # request otherwise leaves half the H200 idle.  Larger batches already
    # provide enough independent KV groups; one wave reduces split partials
    # and the fused norm-gate combine.  The cache remains FP8 throughout.
    source = source.replace(
        "def _attn_port_forward(raw, data, vmath=False):\n",
        "def _attn_decode_fp8_splits(batch, kvg, max_keys, sms):\n"
        "    target = 2 * sms if batch == 1 else sms\n"
        "    ns = -(-target // (batch * kvg))\n"
        "    return min(ns, max(1, -(-max_keys // 64)), 128)\n\n\n"
        "def _attn_port_plan(batch, kvg, max_keys, q_len, window, sms, device, split_mode):\n"
        "    ext = _attn_port_ext()\n"
        "    groups = batch * kvg\n"
        "    per_sm = max(1, int(ext.occupancy(q_len)))\n"
        "    fit = max(1, sms * per_sm // groups)\n"
        "    fit = min(fit, max(1, -(-max_keys // 64)), 16 if window else 512)\n"
        "    if split_mode == 'fp8' and q_len == 1:\n"
        "        target = 2 * sms if batch == 1 else sms\n"
        "        fit = min(fit, max(1, -(-target // groups)))\n"
        "    choices = [cs for cs in range(1, 17) if int(ext.max_cluster(q_len, cs)) >= 1]\n"
        "    best = None\n"
        "    for cs in choices or [1]:\n"
        "        if cs > fit:\n"
        "            continue\n"
        "        nc = min(fit // cs, 32)\n"
        "        while nc > 0 and int(ext.max_cluster(q_len, cs)) < groups * nc:\n"
        "            nc -= 1\n"
        "        if nc:\n"
        "            key = (nc * cs, -nc, cs)\n"
        "            if best is None or key > best[0]:\n"
        "                best = (key, nc * cs, cs)\n"
        "    if best is None:\n"
        "        return 1, 1\n"
        "    return best[1], best[2]\n\n\n"
        "def _attn_port_forward(raw, data, vmath=False, split_mode='auto'):\n",
    )
    source = source.replace(
        "    cs = 1\n    rows = batch * (HKV // 2) * ns * length * 16\n",
        "    ns, cs = _attn_port_plan(batch, HKV // 2, max_keys, length, window,\n"
        "                             _num_sms(device), device, split_mode)\n"
        "    nc = ns // cs\n"
        "    rows = batch * (HKV // 2) * (nc if nc > 1 else 0) * length * 16\n",
        1,
    )
    source = source.replace(
        "    ns = _splits(batch, HKV // 2, max_keys, length, _num_sms(device))\n",
        "",
        1,
    )
    source = source.replace(
        "        return out + [('port_decode', None)]\n",
        "        return out + [('port_decode', None), ('port_decode_fp8_split', None)]\n",
    )
    source = source.replace(
        "    if label == 'port_decode' or label == 'port_verify':\n        return _attn_port_forward(raw, data, vmath=False)\n",
        "    if label == 'port_decode':\n        return _attn_port_forward(raw, data, vmath=False)\n"
        "    if label == 'port_decode_fp8_split':\n        return _attn_port_forward(raw, data, vmath=False, split_mode='fp8')\n"
        "    if label == 'port_verify':\n        return _attn_port_forward(raw, data, vmath=False)\n",
    )
    # The standalone q_len=8 port is the drop-in kernel measured by the lab.
    # On Hopper its four-stage KV ring hides one more page-table/global-load
    # round trip; keep sm_120 on the three-stage portable configuration.
    source = source.replace(
        "cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);",
        "cudaDeviceProp prop; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);\n"
        "  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s)\n"
        "                                  : ((prop.major == 9) ? launch_t<8, 4>(p, s) : launch_t<8, 3>(p, s));",
    )
    source = source.replace(
        "int64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 3>(); }",
        "int64_t occupancy(int64_t q_len) {\n"
        "  if (q_len == 1) return occupancy_t<1, 4>();\n"
        "  cudaDeviceProp prop; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);\n"
        "  return (prop.major == 9) ? occupancy_t<8, 4>() : occupancy_t<8, 3>();\n"
        "}",
    )
    source = source.replace(
        "return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);",
        "if (q_len == 1) return max_cluster_t<1, 4>((int)cs);\n"
        "  cudaDeviceProp prop; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);\n"
        "  return (prop.major == 9) ? max_cluster_t<8, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);",
    )
    source = source.replace(
        "_measure_graph(lambda setting=setting: _fa3_call(q, gate, data, setting), flush, repeats=3)",
        "_measure_graph(lambda q=q, gate=gate, data=data, setting=setting: _fa3_call(q, gate, data, setting), flush, repeats=3)",
    )
    source = source.replace("functions=['attn_forward', 'attn_forward_vmath', 'max_cluster', 'occupancy']",
                            "functions=['attn_forward', 'max_cluster', 'occupancy']")
    source = source.replace("return out + [('port_verify', None), ('port_vmath', None)] + \\",
                            "return out + [('port_verify', None)] + \\")
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
        chunk = _assignment(wgmma, old, new)
        if old == "X_CUDA":
            # Hopper has enough dynamic shared memory for a fourth KV stage;
            # the extra in-flight tile hides page-table/global latency. Keep
            # the portable local stand-in at three stages because sm_120's
            # dynamic-smem limit is smaller.
            chunk = chunk.replace(
                "cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);",
                "cudaDeviceProp prop; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);\n"
                "  cudaError_t e = (q_len == 1) ? launch_t<1, 4>(p, s)\n"
                "                                  : ((prop.major == 9) ? launch_t<8, 4>(p, s) : launch_t<8, 3>(p, s));",
            )
            chunk = chunk.replace(
                "int64_t occupancy(int64_t q_len) { return (q_len == 1) ? occupancy_t<1, 4>() : occupancy_t<8, 3>(); }",
                "int64_t occupancy(int64_t q_len) {\n"
                "  if (q_len == 1) return occupancy_t<1, 4>();\n"
                "  cudaDeviceProp prop; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);\n"
                "  return (prop.major == 9) ? occupancy_t<8, 4>() : occupancy_t<8, 3>();\n"
                "}",
            )
            chunk = chunk.replace(
                "return (q_len == 1) ? max_cluster_t<1, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);",
                "if (q_len == 1) return max_cluster_t<1, 4>((int)cs);\n"
                "  cudaDeviceProp prop; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);\n"
                "  return (prop.major == 9) ? max_cluster_t<8, 4>((int)cs) : max_cluster_t<8, 3>((int)cs);",
            )
        source += chunk + "\n"

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
