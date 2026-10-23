"""H200 SXM parameters for the performance model (docs/reference/hardware.md has the sources).

SPEC values are NVIDIA's peaks; the judge's floors use exactly HBM_BPS and BF16_FLOPS.
The EFF values are the practical model's knobs: achievable rates and fixed costs. Each says
where it came from; they rest on the calibration (`python -m sim.calibrate`, docs/FINDINGS.md)."""

# --- SPEC (NVIDIA H200 datasheet; the judge's TASK.md floors use these two) ---
HBM_BPS = 4.8e12            # HBM3e, bytes/s
BF16_FLOPS = 989e12         # dense tensor-core BF16
FP8_FLOPS = 1979e12         # dense tensor-core FP8 (Hopper has no FP4 tensor cores)
SMS = 132
L2_BYTES = 60 * 2**20       # driver-reported on H200 (DeepGEMM issue #107); H100 SXM has 50 MB
NVLINK_BPS = 450e9          # NVLink 4, per GPU per direction (900 GB/s bidirectional)

# --- EFF: the practical model ---
EFF = {
    # Fraction of HBM_BPS a well-tuned streaming kernel sustains: read-heavy streams reach ~4.45 TB/s
    # on H200 (arXiv 2604.11391) and a copy 4.3 TB/s (SGLang's hopper_bf16_gemv.py note). SGLang's
    # cuBLAS GEMMs fit 2.8 us + bytes / 3.54 TB/s (74%) on the judge (sim.calibrate).
    "bw": 0.90,
    # The first kernel of a timed call: ramp to full bandwidth and the tail, on a cold L2. The
    # judge subtracts the graph launch itself (~4 us), so this is what's left of a lone kernel.
    "ramp_us": 1.5,
    # Each further kernel in the same call: the ~1 us boundary inside a graph (H100, measured) plus
    # its own ramp/tail; 1.5-3 us for a dependent memory-bound kernel without PDL.
    "gap_us": 2.0,
    # TP8 all-reduce on 8xH200: ~5-7 us at 8-32 KB, then NVLS-class ~276 GB/s (arXiv 2606.05951,
    # 2607.08973). Stock uses SGLang's custom all-reduce v2.
    "allreduce_us": 5.5,
    "allreduce_bps": 276e9,
}


def allreduce_us(nbytes: float) -> float:
    """One TP8 all-reduce of nbytes (each GPU's partial sum)."""
    return EFF["allreduce_us"] + nbytes / EFF["allreduce_bps"] * 1e6


def kernels_us(kernels: int) -> float:
    """Fixed cost of a call made of `kernels` kernels."""
    return EFF["ramp_us"] + (kernels - 1) * EFF["gap_us"]


def stream_us(nbytes: float, eff_bw: float | None = None) -> float:
    """Time to stream nbytes from HBM at the achievable rate (no fixed cost)."""
    return nbytes / (HBM_BPS * (EFF["bw"] if eff_bw is None else eff_bw)) * 1e6


def floor_us(nbytes: float, flops: float) -> float:
    """The judge's floor: data at 4.8 TB/s or math at 989 TFLOP/s, whichever is longer."""
    return max(nbytes / HBM_BPS, flops / BF16_FLOPS) * 1e6
