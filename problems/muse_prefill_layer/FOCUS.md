# Prefill problem: H200 lab host only
Serves: G2 (as an H200 lab host only).

The prefill board doesn't count toward the overall score and prefill work is parked with serving (R-20261025-1;
`docs/serving/`). This problem stays useful as a lab host: its prepare receives the decode layer's weight names,
so `tools/h200_lab.py send <payload> --host muse_prefill_layer` times decode-layer stages on real weights.
No lane works on its board.
