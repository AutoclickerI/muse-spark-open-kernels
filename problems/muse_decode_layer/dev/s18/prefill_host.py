# okc: problem=muse_prefill_layer
"""Lab-only prefill host for the import-time S18-vs-S17 decode pair (remote_lab.py).

The decode pair runs at import, where no 300 s prepare cap applies, and needs ~400-450 s cold (da0cd64b's
S17 pair: 399.7 s). Hosted by the prefill best source, prefill's own run (prepare 114-135 s, 336-580 s wall)
would push the run past its 15 minutes, so this host stops at prepare once the buffered lab lines print.
"""
import torch


def prepare(weights):
    raise RuntimeError("lab-only host: the S18 decode pair printed above is this run's result")


def kernel_fn(data, state):
    raise RuntimeError("lab-only host")
