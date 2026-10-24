# Projection lane focus (P7 / G3)

GPU0 owns decode M1-32 and glue. Serving lane owns verify dense M64-256,
with M8 control; its handoff is `dev/verify_streaming/README.md`.


H200 `feb7b287e3464e4a`: M1 streaming GEMV matches its plain-read control
(~2.0-2.5TB/s raw), beats cuBLAS only at M1, loses at every M>=8. This shifts
P7 toward overlap; it is a measured implementation envelope, not a proof.
Best6's small-M custom recipes explicitly disable own-weight prefetch, despite
its global True default; PDL is on. `dev/l2_overlap/PROFILE_MAP.md` maps roles,
activity durations/floors, grouped-profile ambiguities and missing M1 geometry.

`dev/l2_overlap_lab.py`, send key **9367398c624b7796**, prefill host cb8939d0,
priority0.5. Thirteen normal-load/bulk-hint recipes; <=40MiB front/24MiB tail,
captured side-stream forks/joins, unchanged best6 arithmetic/dispatch. Six
M1/16/32 sliding/full cases, three refreshed public streams, cold128MiB AB/BA,
24-round finalists, A/A timing and numerical controls, native activity stages.
Local: all78 coarse recipe/cases,12 finalists,6 stage profiles,5 host tests pass;
12.0s cached payload in an80s share. Raw/router FP32 gate1e-4, mixed-BF16
layer0.3%, exact A/B IDs, independent V9/attention/norm gates. Native sm90a:
real128-bit loads + UBLKPF.L2, no spills,44/18 registers; no architecture skip.
Best7 projection cell `{'l2_prefetch': recipe_id}` is implemented and defaults
off. H200 whole-layer gains decide each decode row; local timing does not.
`dev/l2_overlap/read_receipt.py <job>` reads only landed SHA-matched evidence.

Prior glue+verify dense receipt **5535d2f171b34d72**: all fused/WGMMA variants
lost; none promoted. Historical unsafe overlap
stays deleted. New transport uses explicit CUDA global addresses and executes
both paths locally. No prepared NVFP4 expansion or L2 persistence is used.
