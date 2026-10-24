"""Crossovers, Marlin dead storage, and tiny cached-prefix graph regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.ops.muse_spark_v12 import gemm_ag
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase


class UpstreamPolicy(CustomTestCase):
    def test_projection_reservation_covers_width_specific_replays(self):
        # The previous shared M128 bound missed pooler M256 and selected a
        # measured losing unpooler M128. Reservation must cover each width.
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            for world in (4, 8):
                self.assertEqual(gemm_ag.projection_max_tokens(world, 4096, "cuda"), 512)
                self.assertEqual(gemm_ag.projection_max_tokens(world, 8192, "cuda"), 64)
            self.assertEqual(gemm_ag.projection_max_tokens(2, 8192, "cuda"), 8)

    def test_tiny_prefix_can_replay_without_admitting_large_padding(self):
        f = PrefillCudaGraphRunner._padding_within_budget
        self.assertTrue(f(1, 4, [4, 16]))
        self.assertTrue(f(1, 16, [16, 32]))
        self.assertFalse(f(1, 32, [32, 64]))
        self.assertFalse(f(5, 16, [4, 16]))
        self.assertFalse(f(100, 2816, [2816, 3072, 8192]))
        self.assertTrue(f(8, 16, [4, 16]))

    def test_marlin_omits_only_dead_scale_copies(self):
        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        method.quant_config = SimpleNamespace(group_size=16, get_name=lambda: "modelopt_fp4",
                                            is_checkpoint_nvfp4_serialized=True)
        method.enable_flashinfer_trtllm_moe = False
        method._moe_runner_backend = MoeRunnerBackend.MARLIN
        layer = torch.nn.Module()
        layer.num_local_experts = layer.num_experts = 2
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        with get_context().override_server_args(moe_runner_backend="marlin"):
            method.create_weights(layer, 2, 128, 128, torch.bfloat16,
                                  weight_loader=lambda *a, **kw: None)
        self.assertIsNone(layer.w13_blockscale_swizzled)
        self.assertIsNone(layer.w2_blockscale_swizzled)
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale.shape, (2, 256, 8))
        self.assertIn("w13_weight_scale_2", layer._parameters)
        self.assertIn("w2_weight_scale_2", layer._parameters)


if __name__ == "__main__":
    unittest.main()
