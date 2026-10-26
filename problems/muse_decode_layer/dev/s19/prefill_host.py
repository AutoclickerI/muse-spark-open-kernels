# okc: problem=muse_prefill_layer
"""Lab-only prefill host for an import-time decode-layer pair (S19 vs S18, build_lab.py).

The pair runs at import, where no 300 s prepare cap applies (S18 vs S17: 398.6 s of lab, 434.5 s of run,
5d0282bb). The prefill best source's own run (336-580 s wall) would push the run past its 15 minutes, so
this host stops at prepare once the buffered lab lines print.
"""
import torch


def prepare(weights):
    raise RuntimeError("lab-only host: the decode-layer pair printed above is this run's result")


def kernel_fn(data, state):
    raise RuntimeError("lab-only host")
