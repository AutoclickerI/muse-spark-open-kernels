"""Portable ABI audit for the SM90/topology gates in serving/patch.diff.

The workstation is SM120, so this test forces the Python gates to report SM90
and replaces each CUDA entry point with a small torch implementation having the
same argument and return ABI.  It is intentionally separate from the numerical
kernel tests: a tuple-versus-tensor or cache-location mistake must fail before
any Hopper cubin is available.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.kernels.ops.muse_spark_v12 import all_gather_norm_residual as agr
from sglang.kernels.ops.muse_spark_v12 import gemm_ag as ga
from sglang.kernels.ops.muse_spark_v12 import norm_residual as nr
from sglang.srt.layers import muse_spark_fp4_moe as fp4
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.models import muse_spark as model


@contextmanager
def sm90():
    old = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *args, **kwargs: (9, 0)
    try:
        yield
    finally:
        torch.cuda.get_device_capability = old


class PortableNorm:
    def run_std(
        self, branch, stream, output, new_stream, router, post_w, pre_w, alpha, beta,
        post_eps, pre_eps, pre_bf16
    ):
        x = branch.float()
        if post_w is not None:
            x = x * post_w.float()
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + post_eps)
        output.copy_(y.to(torch.bfloat16))
        new_stream.copy_(stream + output.float())


class PortableAG:
    def run(self, obj, local, stream, output, new_stream, post_w, pre_w, alpha, beta, post_eps, pre_eps):
        output.copy_(obj.full_output)
        new_stream.copy_(stream + output.float())


class PortableGather:
    def run(self, obj, local, bias, residual, out):
        out.copy_(obj.full_output)


def test_sm90_policies_and_standins() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    with sm90():
        assert agr._supports_clusters()
        assert nr._max_cluster_size() == 8
        assert [nr.norm_residual_launch_config(8192, m)[0] for m in (1, 8, 16, 32, 64, 128)] == [8, 8, 8, 4, 2, 1]

        branch = torch.randn(8, 8192, device=device, dtype=torch.bfloat16)
        stream = torch.randn(8, 8192, device=device, dtype=torch.float32)
        old_norm = nr._jit_module
        nr._jit_module = lambda *args, **kwargs: PortableNorm()
        try:
            out, new_stream, router = nr.fused_norm_residual(
                branch, stream, 1e-5, 1e-5, cluster_size=None
            )
        finally:
            nr._jit_module = old_norm
        assert out.dtype == torch.bfloat16 and new_stream.dtype == torch.float32 and router is None
        assert out.shape == branch.shape and new_stream.shape == stream.shape

        ag_local = torch.randn(8, 8192, device=device, dtype=torch.bfloat16)
        ag_stream = torch.randn(8, 8192, device=device, dtype=torch.float32)
        ag_obj = SimpleNamespace(full_output=ag_local)
        ag_comm = SimpleNamespace(world_size=8, obj=ag_obj)
        old_ag = agr._jit_module
        agr._jit_module = lambda *args, **kwargs: PortableAG()
        try:
            ag_out, ag_new_stream = agr.all_gather_norm_residual(
                ag_comm, ag_local, ag_stream, 1e-5, 1e-5
            )
        finally:
            agr._jit_module = old_ag
        assert ag_out.shape == ag_local.shape and ag_new_stream.shape == ag_stream.shape
        assert ag_out.dtype == torch.bfloat16 and ag_new_stream.dtype == torch.float32

        local = torch.randn(8, 2048, device=device, dtype=torch.bfloat16)
        full = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)
        obj = SimpleNamespace(full_output=full)
        comm = SimpleNamespace(
            world_size=8, rank=0, has_multicast=True, disabled=False,
            max_push_size=1 << 30, config=SimpleNamespace(num_push_blocks=8), obj=obj,
        )
        w = torch.randn(4096, 2048, device=device, dtype=torch.bfloat16)
        assert ga.projection_max_tokens(8, 4096, device) == 512
        assert ga.projection_max_tokens(8, 8192, device) == 64
        old_gather = ga._jit_module
        old_linear = __import__(
            "sglang.srt.layers.quantization.unquant", fromlist=["bf16_gemm_dispatch"]
        ).bf16_gemm_dispatch
        ga._jit_module = lambda *args, **kwargs: PortableGather()
        unquant = __import__("sglang.srt.layers.quantization.unquant", fromlist=["bf16_gemm_dispatch"])
        unquant.bf16_gemm_dispatch = lambda x, weight, bias: torch.nn.functional.linear(x, weight, bias)
        try:
            got = ga.gemm_ag(comm, local, w)
        finally:
            ga._jit_module = old_gather
            unquant.bf16_gemm_dispatch = old_linear
        assert torch.equal(got, full)

        old_cap = model.torch.cuda.is_current_stream_capturing
        old_cpu, old_forward = model._is_cpu, model.get_forward
        model.torch.cuda.is_current_stream_capturing = lambda: False
        model._is_cpu = False
        model.get_forward = lambda: SimpleNamespace(sp_active=False)
        try:
            assert not model._use_row_split(1, 2048)
            assert not model._use_row_split(8, 1023)
            assert model._use_row_split(8, 1024)
        finally:
            model.torch.cuda.is_current_stream_capturing = old_cap
            model._is_cpu, model.get_forward = old_cpu, old_forward


def test_sm90_backend_and_fa3_abi() -> None:
    with sm90():
        old_cuda, old_cap, old_backend = fp4.is_cuda, fp4.get_device_capability, fp4.get_moe_runner_backend
        fp4.is_cuda = lambda: True
        fp4.get_device_capability = lambda: (9, 0)
        fp4.get_moe_runner_backend = lambda: MoeRunnerBackend.AUTO
        try:
            assert fp4._resolve_moe_runner_backend() is MoeRunnerBackend.MARLIN
        finally:
            fp4.is_cuda, fp4.get_device_capability, fp4.get_moe_runner_backend = old_cuda, old_cap, old_backend

        class FakeFA3:
            fa_impl_ver = 3
            use_mla = False
            fa_skip_kv_cache = False
            data_type = torch.float8_e4m3fn

        class FakeMode:
            def is_decode(self):
                return False

            def is_target_verify(self):
                return True

        class FakePool:
            dtype = torch.float8_e4m3fn
            layers_mapping = [(0, False)]

            def get_key_buffer(self, layer):
                return torch.empty(4096, 128, device="cuda", dtype=self.dtype)

            def get_value_buffer(self, layer):
                return torch.empty(4096, 128, device="cuda", dtype=self.dtype)

        attn = SimpleNamespace(k_scale=None, v_scale=None)
        obj = object.__new__(model.MuseSpark1xAttention)
        obj.layer_idx = 0
        obj.attn = attn
        old_backend_cls, old_get_backend, old_get_pool, old_cp = (
            model.FlashAttentionBackend, model.get_attn_backend,
            model.get_token_to_kv_pool, model.is_cp_active,
        )
        model.FlashAttentionBackend = FakeFA3
        model.get_attn_backend = lambda: FakeFA3()
        model.get_token_to_kv_pool = lambda: FakePool()
        model.is_cp_active = lambda _: False
        try:
            fb = SimpleNamespace(forward_mode=FakeMode(), dcp_kv_mask=None,
                                 out_cache_loc=torch.arange(8, device="cuda", dtype=torch.int64) + 17)
            kv = obj._fused_kv_store(fb)
        finally:
            model.FlashAttentionBackend, model.get_attn_backend = old_backend_cls, old_get_backend
            model.get_token_to_kv_pool, model.is_cp_active = old_get_pool, old_cp
        assert isinstance(kv, tuple) and len(kv) == 3
        assert kv[0].dtype == torch.float8_e4m3fn and kv[2].dtype == torch.int64
        assert kv[2].is_contiguous()


if __name__ == "__main__":
    test_sm90_policies_and_standins()
    test_sm90_backend_and_fa3_abi()
    print("SM90 portable glue audit: PASS")
