# Integration handoff: decode-1 direct projection partials

Apply `patch_partial_port.py` (SHA-256 **d454609d3d07eb24da7c5f47aa1b988959dca25572867a5bfe6176069ccb6eac**) onto frozen
`problems/muse_decode_layer/cand_r3_best7_s11_final.py` (**acda6c62**).
Its default rows are decode-1 sliding and full only; exact sincos, one attention launch.
Generated candidate: `out/s11_partial_port_decode1.py` (**07157537a2a19a43ddaba8ea8f0cecadb42e6fdbe284240786ed26e1412f943e**).

H200 whole-layer pair **c114c35e787f46d1**, 16 seeds x8 cold rotating replays, same inputs/state:
- sliding: **0.96960**, SE .00115, **-2.25 us** per layer; A:A .99941.
- full: **0.97415**, SE .00119, **-2.21 us**; A:A 1.00235.
- 46/16 weighted decode-1 ratio **0.97091**. Verify rows are unchanged.

All diagnostic output gates pass. The all-decode experimental host passed official H200 correctness and
timed at 1.72965x. Normal and forced local screens each passed 13 tests, 12 fresh-input graphs and 24 static
checks for that same kernel. Default row restriction removes the unselected dispatches; it changes no kernel.

Reproduce: `python3 problems/muse_attention_decode/dev/vq8/patch_partial_port.py <S11-final> <output>`.
Dependencies beside it: `partial_port.py` (extracts/pins S11's port and emits the cell).
Kernel code consumes `(P,S)` FP32 projection partials or BF16 one-split, rounds at the original BF16 boundary,
retains cache append, all keys, cluster combine, norm/gate and the gate-split ungated path.

The attention lane now targets structural sliding latency using critical-path increments in `dev/timeline`.
