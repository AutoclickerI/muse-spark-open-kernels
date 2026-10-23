"""Muse Spark Open's shapes, per GPU, as served on 8xH200 (TP8 attention and dense, EP8 experts).

From SGLang's config (serving/sglang/python/sglang/srt/configs/muse_spark.py, commit a55ab7d5),
the problems' reference.py files, and the DFlash2 drafter's config.json on Hugging Face
(RadixArk/Muse-Spark-Open-DFlash2). Byte counts are what one GPU must read per call."""

HIDDEN = 8192
LAYERS = 62
SLIDING_LAYERS, FULL_LAYERS = 46, 16   # the problems' load weights; HF layer_types is gated
VOCAB = 202048
TP = EP = 8
HEAD_DIM = 64
HQ, HKV = 128 // TP, 16 // TP          # 16 query, 2 KV heads per GPU
Q_SIZE, KV_SIZE = HQ * HEAD_DIM, HKV * HEAD_DIM
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE    # q, k, v, output gate: 2304
WINDOW = 2048
D_MOE = 4096                           # pooler/unpooler width = expert width = expert intermediate
N_EXPERTS, TOP_K = 256, 8
LOCAL_EXPERTS = N_EXPERTS // EP        # 32
UNPOOL_SLICE, SLICED_MAX_TOKENS = HIDDEN // 8, 8
NVFP4_BYTES_PER_WEIGHT = 0.5 + 1 / 16  # E2M1 + one E4M3 scale per 16
EXPERT_BYTES = 3 * D_MOE * D_MOE * NVFP4_BYTES_PER_WEIGHT   # 28.3 MB: w13 (gate, up) and w2
KV_BYTES_PER_TOKEN = 2 * KV_SIZE       # K and V, FP8: 256 B per token per layer per GPU

# Dense BF16 weights per layer per GPU (bf16_proj's LAYERS); the router is FP32.
QKV_GATE_BYTES = QKV_SIZE * HIDDEN * 2
O_BYTES = HIDDEN * Q_SIZE * 2
POOLER_BYTES = D_MOE * HIDDEN * 2      # replicated on every rank
ROUTER_BYTES = N_EXPERTS * HIDDEN * 4


def unpooler_bytes(tokens: int) -> float:
    """Up to 8 tokens each GPU computes only its 1/8 of the unpooler's columns."""
    return (UNPOOL_SLICE if tokens <= SLICED_MAX_TOKENS else HIDDEN) * D_MOE * 2


def dense_bytes(tokens: int) -> float:
    return QKV_GATE_BYTES + O_BYTES + POOLER_BYTES + ROUTER_BYTES + unpooler_bytes(tokens)


def kv_bytes(batch: int, ctx: float, layer: str) -> float:
    return batch * (min(ctx, WINDOW) if layer == "sliding" else ctx) * KV_BYTES_PER_TOKEN


# Vocab-parallel LM head: each GPU holds VOCAB/8 rows.
LM_HEAD_BYTES = VOCAB // TP * HIDDEN * 2

# DFlash2 drafter (Qwen3-style, 5 layers, non-causal block of 8, sliding window 2048), TP8.
DRAFT = {"layers": 5, "hidden": 8192, "inter": 28672, "heads": 64, "kv_heads": 8, "head_dim": 128,
         "block": 8, "context_layers": 5}


def drafter_bytes() -> float:
    """Weights one GPU reads for one drafter forward (TP8), plus a second pass over the target's
    vocab-parallel LM head: DFlash2 takes its top-k candidates through it (models/dflash.py)."""
    d = DRAFT
    attn = d["hidden"] * (2 * d["heads"] + 2 * d["kv_heads"]) * d["head_dim"]   # q, k, v, o
    mlp = 3 * d["hidden"] * d["inter"]
    fc = d["context_layers"] * d["hidden"] * d["hidden"]    # concat of 5 target layers -> hidden
    return (d["layers"] * (attn + mlp) + fc) * 2 / TP + LM_HEAD_BYTES
