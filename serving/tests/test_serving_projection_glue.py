"""Force TP8 multicast Python integration at exact full model shapes on GPU0."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from sglang.srt.layers import muse_spark_serving as adapter
from sglang.srt.layers.muse_proj import MuseProjection
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import muse_spark as model


class Experts(torch.nn.Module):
    post_expert_norm = None

    def forward(self, hidden, *args, **kwargs):
        return hidden


def main():
    root = torch.nn.Module()
    attn = model.MuseSpark1xAttention.__new__(model.MuseSpark1xAttention)
    torch.nn.Module.__init__(attn)
    attn.o_proj = torch.nn.Module()
    attn.o_proj.weight = torch.nn.Parameter(
        torch.randn(8192, 1024, device="cuda", dtype=torch.bfloat16) / 32,
        requires_grad=False,
    )
    attn.o_proj.quant_method = UnquantizedLinearMethod()
    root.attn = attn
    moe = model.MuseSpark1xMoE.__new__(model.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    root.moe = moe
    moe.tp_size = 8
    moe.moe_tp_size = 1
    moe.hidden_size = 8192
    moe.expert_hidden_size = 4096
    moe.router_in_fp32 = True
    moe.nvfp4_checkpoint = False
    moe.top_k = 8
    moe.pre_expert_norm = None
    moe.pre_expert_proj = torch.nn.Linear(
        8192, 4096, bias=False, device="cuda", dtype=torch.bfloat16
    )
    moe.post_expert_proj = torch.nn.Linear(
        4096, 8192, bias=False, device="cuda", dtype=torch.bfloat16
    )
    moe.pre_expert_proj.weight.requires_grad_(False)
    moe.post_expert_proj.weight.requires_grad_(False)
    moe.gate = model.MuseSpark1xRouter(8192, 256, torch.float32).cuda()
    moe.gate.weight.data.normal_(std=0.01)
    moe.experts = Experts()
    comm = SimpleNamespace(world_size=8, rank=3, has_multicast=True, disabled=False)
    group = SimpleNamespace(ca_comm=comm)
    prepared = []

    def construct(weight, kind, **kwargs):
        prepared.append(
            (
                kind,
                tuple(weight.shape),
                kwargs["tokens"] if "tokens" in kwargs else None,
            )
        )
        return MuseProjection(weight, kind, tokens=(), budget_s=0)

    gate = lambda x, w, k: x.is_cuda and 0 < x.shape[0] <= 256
    with (
        patch("sglang.srt.distributed.get_tp_group", return_value=group),
        patch.object(adapter, "projection_supported", side_effect=gate),
        patch.object(adapter, "MuseProjection", side_effect=construct),
        patch.object(adapter, "projection_max_tokens",
                     side_effect=lambda world, n, device: 512 if n == 4096 else 64),
    ):
        adapter.prepare_projections(root)
    assert isinstance(attn.o_proj.quant_method, adapter.ProjectionMethod)
    assert ("pooler", (4096, 8192), ()) in prepared
    assert ("unpooler", (8192, 4096), (128, 256)) in prepared
    assert ("pooler", (512, 8192), (8, 16, 32, 64, 128, 256)) in prepared
    assert ("unpooler", (1024, 4096), (8, 16, 32, 64)) in prepared
    gathers = []
    current = {}

    def gather(comm, local, full):
        weight, x = current["weight"], current["x"]
        expected = F.linear(x, weight.chunk(8)[3])
        assert (
            float((local.float() - expected.float()).norm() / expected.float().norm())
            < 0.01
        ), "wrong rank slice"
        full.copy_(F.linear(x, weight))
        gathers.append(tuple(local.shape))
        return full

    route = lambda _, logits, k, *args, **kwargs: (
        torch.ones(logits.shape[0], 8, device="cuda"),
        torch.zeros(logits.shape[0], 8, device="cuda", dtype=torch.int32),
        None,
    )
    with (
        patch.object(model, "get_tp_group", return_value=group),
        patch.object(model, "gemm_ag_supported", return_value=True),
        patch.object(model, "projection_all_gather", side_effect=gather),
        patch.object(model, "_use_split_bf16_router", return_value=False),
        patch.object(model, "get_global_experts_capturer", return_value=None),
        patch.object(moe, "_routing_op", return_value=route),
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        for m in (8, 32, 128):
            x = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
            current.update(weight=moe.pre_expert_proj.weight, x=x)
            got = moe._forward_experts(x)
            expected = F.linear(x, moe.pre_expert_proj.weight)
            assert torch.equal(got, expected)
            current.update(weight=moe.post_expert_proj.weight, x=got)
            with patch.object(moe, "_defers_unpool_gather", return_value=False):
                full = moe._reduce_and_unpool(got, already_reduced=True)
                assert torch.equal(full, F.linear(got, moe.post_expert_proj.weight))
            before = len(gathers)
            with patch.object(moe, "_defers_unpool_gather", return_value=True):
                local = moe._reduce_and_unpool(got, already_reduced=True)
                expected_local = F.linear(got, moe.post_expert_proj.weight.chunk(8)[3])
                assert (
                    float(
                        (local.float() - expected_local.float()).norm()
                        / expected_local.float().norm()
                    )
                    < 0.01
                )
            assert len(gathers) == before, "deferred gather ran early"
    assert len(gathers) == 6
    print(
        "TP8 exact-shape prepare, pooler gather, unpooler gather/defer: passed (local L2<1%, full maxdiff0)",
        flush=True,
    )


if __name__ == "__main__":
    main()
