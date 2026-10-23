"""Harness-only import hook: serve SGLang's prebuilt Rust extensions from the image.

The image's own SGLang (/sgl-workspace/sglang-muse-spark, = a55ab7d5 plus build artifacts)
ships the PyO3 extensions built from the same rust/ sources our checkouts carry; a bare source
checkout has none. Without them SGLang either picks the Python TreeCore or, when `cargo` answers
its version probe, picks the Rust TreeCore and then fails to load it (SGLANG_RUST_BUILD_MODE=never
in the image), and the probe itself makes rustup download a toolchain. Mapping the modules to the
image's builds makes both trees run the bundled Rust TreeCore, as an installed SGLang does.

The normal finders run first, so an in-tree build would still win. Set MUSE_HARNESS_NO_RUST_HOOK=1
to disable.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_SRT = "/sgl-workspace/sglang-muse-spark/python/sglang/srt"
_SUFFIX = ".cpython-312-x86_64-linux-gnu.so"
_MODULES = {
    "sglang.srt.mem_cache.rust_tree_core.mem_cache": f"{_SRT}/mem_cache/rust_tree_core/mem_cache{_SUFFIX}",
    "sglang.srt.rust_extensions._grpc": f"{_SRT}/rust_extensions/_grpc{_SUFFIX}",
    "sglang.srt.rust_extensions._multimodal": f"{_SRT}/rust_extensions/_multimodal{_SUFFIX}",
    "sglang.srt.rust_extensions._server": f"{_SRT}/rust_extensions/_server{_SUFFIX}",
}


class _ImageRustExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        so = _MODULES.get(name)
        if so is None or not os.path.exists(so):
            return None
        return importlib.util.spec_from_file_location(
            name, so, loader=importlib.machinery.ExtensionFileLoader(name, so))


if os.environ.get("MUSE_HARNESS_NO_RUST_HOOK") != "1":
    sys.meta_path.append(_ImageRustExtensions())

# Chain to the interpreter's own sitecustomize, which this module shadows.
_system = f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/sitecustomize.py"
if os.path.exists(_system):
    with open(_system) as _f:
        exec(compile(_f.read(), _system, "exec"))


# ------------------------------------------------------------------ weight-load check
# With MUSE_HARNESS_LOAD_CHECK=<dir>, every parameter of the target (MuseSparkForConditionalGeneration)
# and the drafter (DFlash*DraftModel) is filled with a sentinel (NaN, or 0xA5 bytes) right before
# load_weights and inspected right after it: a parameter still holding the sentinel was never
# written by SGLang's loader. Each process writes <dir>/<class>-<pid>.json, then puts the original
# values back so the server runs exactly as without the check.
_LOAD_CHECK = os.environ.get("MUSE_HARNESS_LOAD_CHECK")
_PATCH_TARGETS = {
    "sglang.srt.models.muse_spark_omni": "MuseSparkForConditionalGeneration",
    "sglang.srt.models.dflash": "DFlashDraftModel",
}


def _sentinel_wrap(cls, out_dir):
    import json as _json

    import torch as _torch

    orig = cls.load_weights

    def _fill(t):
        if t.dtype in (_torch.float8_e4m3fn, _torch.float8_e5m2):
            t.view(_torch.uint8).fill_(0x7F)      # NaN in E4M3
        elif t.is_floating_point():
            t.fill_(float("nan"))
        elif t.element_size() == 1:
            t.view(_torch.uint8).fill_(0xA5)
        else:
            t.fill_(-12345)

    def _unwritten(t):
        if t.is_floating_point():
            if t.dtype in (_torch.float8_e4m3fn, _torch.float8_e5m2):
                t = t.view(_torch.uint8)
                return (t == 0x7F).float().mean().item() if t.numel() else 0.0
            return _torch.isnan(t).float().mean().item() if t.numel() else 0.0
        if t.element_size() == 1:
            return (t.view(_torch.uint8) == 0xA5).float().mean().item() if t.numel() else 0.0
        return (t == -12345).float().mean().item() if t.numel() else 0.0

    def load_weights(self, weights):
        tensors = dict(self.named_parameters())
        tensors.update({n: b for n, b in self.named_buffers()
                        if n.endswith("e_score_correction_bias") and b is not None})
        saved = {n: t.detach().clone() for n, t in tensors.items()}
        with _torch.no_grad():
            for t in tensors.values():
                _fill(t.data)
        # Replay the cold loader with a complementary byte sentinel. A single fixed
        # uint8 sentinel collides naturally with roughly 1/256 of FP4 bytes.
        weights = list(weights)
        result = orig(self, iter(weights))
        first = {n: t.detach().clone() for n, t in tensors.items()}
        with _torch.no_grad():
            for n, t in tensors.items():
                if t.dtype == _torch.uint8:
                    t.copy_(first[n].bitwise_not())
        # Only uint8 parameters need a second pass; floating NaNs cannot collide
        # with the finite checkpoint. Replaying also recomputes cached norm gains.
        if any(t.dtype == _torch.uint8 for t in tensors.values()):
            result = orig(self, iter(weights))
        report = {}
        with _torch.no_grad():
            for n, t in tensors.items():
                frac = ((t != first[n]).float().mean().item() if t.dtype == _torch.uint8
                        else _unwritten(t.data)) if t.data.shape == saved[n].shape else None
                report[n] = {"shape": list(saved[n].shape), "dtype": str(saved[n].dtype),
                             "unwritten_fraction": frac}
                if frac == 1.0:            # never loaded: restore the constructor's value
                    t.data.copy_(saved[n])
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{type(self).__name__}-{os.getpid()}.json")
        with open(path, "w") as f:
            _json.dump(report, f, indent=1)
        return result

    cls.load_weights = load_weights


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in _PATCH_TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _sentinel_wrap(getattr(module, _PATCH_TARGETS[name]), _LOAD_CHECK)

        spec.loader.exec_module = exec_and_patch
        return spec


if _LOAD_CHECK:
    sys.meta_path.insert(0, _PatchAfterImport())

if os.environ.get("MUSE_HARNESS_TRACE"):
    import observe

# Diagnostic only: PyTorch fills empty floating allocations with NaNs when
# deterministic algorithms are enabled. This can expose reads before writes.
if os.environ.get('MUSE_HARNESS_POISON_EMPTY') == '1':
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
