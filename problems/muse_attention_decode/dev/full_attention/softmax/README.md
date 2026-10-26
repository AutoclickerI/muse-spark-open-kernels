# softmax

S14 full-verify softmax issue study; no variant promoted.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [audit_sass.py](audit_sass.py): Count the isolated actual softmax/rescale probe and its source phases on sm90a.
- [build.py](build.py): S14 softmax issue study: split SFU/FMA exponentials, tensor-core sums and WG staggering.
- [build_lab.py](build_lab.py): Freeze one batched full-attention payload, including independent native tensor-sum probes.
- [build_patch.py](build_patch.py): Generate a selected softmax-only patch onto exact S14 final; selection requires H200 evidence.
- [build_ping.py](build_ping.py): Alternate two consumer WGs: QK/PV for one, softmax for the other; producer unchanged.
- [check.py](check.py): Probe math then full attention/cache/capture and strided FP32 projection ABI.
- [checks.py](checks.py): Independent FP32 exp/softmax, packed-P layout and rescale checks for the lab and local screen.
- [gap.cu](gap.cu): Device source or payload template used by retained code.
- [gap.py](gap.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [lab.py.inc](lab.py.inc): Device source or payload template used by retained code.
- [mix25.cu](mix25.cu): Device source or payload template used by retained code.
- [mix25.py](mix25.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [mix50.cu](mix50.cu): Device source or payload template used by retained code.
- [mix50.py](mix50.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [mixgap.cu](mixgap.cu): Device source or payload template used by retained code.
- [mixgap.py](mixgap.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [ping.cu](ping.cu): Device source or payload template used by retained code.
- [ping.py](ping.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [ping25.cu](ping25.cu): Device source or payload template used by retained code.
- [ping25.py](ping25.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [ref.cu](ref.cu): Device source or payload template used by retained code.
- [ref.py](ref.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [softmax_mix_lab.py](softmax_mix_lab.py): Self-contained H200 measurement payload.
- [tc25.cu](tc25.cu): Device source or payload template used by retained code.
- [tc25.py](tc25.py): Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a).

Handoff: [build_patch.py](build_patch.py), SHA256 `3f13695b454606e23d86fec43d8c454a3983973375894c946cf24630981ff5ee`, applies onto S14 f1c77ce6, VQ8 CUDA c262e4c5.
Base source: [S14 final](../../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

Reset result: **457f3c73b3214534** (wrapper03086413), complete native gates/probes,
3seeds ×16cold rotating replays. Ratios16/32: mix25 **1.05012/1.04610**, mix50
**1.08436/1.07352**, gap **1.00004/1.00126**, mixgap **1.05200/1.04865**,
tc25 **1.03473/1.03452**, ping **1.15929/1.17091**, ping25 **1.22654/1.21961**.
A:A **0.99941/0.99947**, ranges **0.99562–1.00164 /0.99608–1.00154**.
No winner or P4 promotion. The mixed exps increase issue, and the tested ping-pong
adds synchronization. TC25 reduces scalar alpha/sum work, but its standalone
WGMMA group/wait and polynomial confound remain. The follow-up targets S15 N128:
keep native MUFU, carry the denominator as four PV accumulator registers and
issue P×ones in the existing PV group. It retains exact FP8 operands, max/rescale
logic, all keys and the stage4 ring. `../pvsum_build.py` builds that candidate;
independent denominator/rescale probes and native SASS gates precede any send.
`reset_result.json` records the exact receipt and clocks; FINDINGS has one row per recipe.
