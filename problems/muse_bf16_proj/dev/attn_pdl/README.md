# Attention early release + O weight prefetch (P7 glue)

No S6 attention kernel runs `griddepcontrol.launch_dependents`, so a PDL successor can start only when
attention exits. S6's M=1 O cell (GF config6) loads its whole 16.8 MB slice into registers before its
`griddepcontrol.wait`, but without the release that prefetch never overlaps attention (the PDL lab
7426e17c measured own-weight prefetch with this release missing; its null result does not close this).

- `lab_body.py` + `build.py` -> `prof/lab/attn_pdl_o.py` (S6's exact attn_main and GF sources):
  chain attention -> O -> PDL consumer, attention as is vs with the release at entry, O = cuBLAS / GF6 /
  GF M16-32 configs / SRG (whole O slice in shared memory via bulk copies before the wait).
  H200 `4d995f13ad694c54`: release + GF6 at decode 1 sliding 21.81 -> 20.86 us (0.9626), full 30.45 -> 28.38
  (0.9300); SRG -1.17 us at sliding, worse at full; decode 16 full release + GF4 0.9908 vs cuBLAS; decode 32 none.
- `trace_local.py`: local globaltimer trace of GF CTA entry vs wait. sm_120: with the release, O CTAs
  enter at 0.26 us (median) and finish pre-wait loads by 2.3 us; O ends 4.1 us after attention vs 5.6 us.
- `patch_trigger.py`: applies the one-line release to a candidate (in `cand_r3_best7_s7_p7c.py`).
  Only scheduling changes: every PDL successor waits before reading attention's output.
- `resources.py`: sm_90a ptxas usage. attn_main<1,4> 110 regs x 128 threads, 74 KB smem; GF config6
  104-139 regs x 128 threads; SRG 53 regs x 256 threads + 128 KB smem.

Open: at M=16/32 the prepare tuner picks cuBLAS for O standalone, which cannot overlap; a PDL O there
must be chosen by whole-layer timing. The 'layer' attention ext (CUDA_SRC attn_main + attn_combine,
plain launches) needs the release in both kernels and a PDL launch of attn_combine to reach O.
