"""H200 SXM parameters for the performance model (docs/CEILING.md has the sources).

SPEC values are NVIDIA's peaks; the judge's floors use exactly HBM_BPS and BF16_FLOPS.
The EFF values are the practical model's knobs: achievable rates and fixed costs. Each says
where it came from, and the calibration in docs/CEILING.md is what they rest on."""

# --- SPEC (NVIDIA H200 datasheet; the judge's TASK.md floors use these two) ---
HBM_BPS = 4.8e12            # HBM3e, bytes/s
BF16_FLOPS = 989e12         # dense tensor-core BF16
FP8_FLOPS = 1979e12         # dense tensor-core FP8 (Hopper has no FP4 tensor cores)
SMS = 132
L2_BYTES = 50 * 2**20       # GH100 SXM: 50 MB
NVLINK_BPS = 450e9          # NVLink 4, per GPU per direction (900 GB/s bidirectional)

# --- EFF: the practical model ---
EFF = {
    # Fraction of HBM_BPS a well-tuned streaming kernel sustains. Measured: SGLang's cuBLAS GEMMs
    # fit 2.8 us + bytes / 3.54 TB/s (74%) on the judge (runs/muse_bf16_proj, sim.calibrate).
    "bw": 0.88,
    # The first kernel of a timed call: ramp to full bandwidth and the tail, on a cold L2. The
    # judge subtracts the graph launch itself (~4 us), so this is what's left of a lone kernel.
    "ramp_us": 1.5,
    # Each further kernel in the same call: the launch gap inside a graph plus its own ramp/tail.
    "gap_us": 2.0,
    # One small all-reduce across 8 GPUs (custom one-shot, decode-sized messages).
    "allreduce_us": 8.0,
}


def kernels_us(kernels: int) -> float:
    """Fixed cost of a call made of `kernels` kernels."""
    return EFF["ramp_us"] + (kernels - 1) * EFF["gap_us"]


def stream_us(nbytes: float, eff_bw: float | None = None) -> float:
    """Time to stream nbytes from HBM at the achievable rate (no fixed cost)."""
    return nbytes / (HBM_BPS * (EFF["bw"] if eff_bw is None else eff_bw)) * 1e6


def floor_us(nbytes: float, flops: float) -> float:
    """The judge's floor: data at 4.8 TB/s or math at 989 TFLOP/s, whichever is longer."""
    return max(nbytes / HBM_BPS, flops / BF16_FLOPS) * 1e6
