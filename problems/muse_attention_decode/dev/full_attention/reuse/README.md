# K-major receipt response: reclaim score-buffer occupancy

Base S15 60254883. Job1bab0b2d0dc14931 established native correctness but the
initial K-major remap costs19.4/24.3% whole-layer time at full decode16/32.
Its6144Bstatic+73984Bdynamic shared uses80128B/CTA: three exceed H200's233472B.
This occupancy hypothesis is distinct from the confirmed native instruction/layout.

`../rebuild_kmajor_reuse.py` retains both heads' completed QK scores briefly in
registers, synchronizes all128threads, and uses the consumed cached K tile as
the shared score matrix. PV reads V, and the prefetch/new-key ring slots are
distinct. Every query, cache key, gather, split, P precision, PV operation and
new-token path stays the same. This removes4096Bstatic shared, with one extra
CTA barrier. Native O3 shows114regs,zero spills,2048Bstatic shared and4native
QGMMA64x8x32 instructions, no C75 serialization warning. Three CTAs now fit by
shared bytes (3*76032=228096); the lab records real native API occupancy too.

The new payload uses S15/A:A/old K-major/reuse at12shapes,8fresh seeds
128107923+7919*i,8rotated cold replays and488public TASK checks, including
all13official fixtures eagerly/after replay. Native layout probes precede
TASK/timing. `build_rebuild_lab.py --reuse` generates it entirely from tracked
public sources; no input-derived output is kept. `screen_rebuild_lab.py --reuse`
and `--reuse --forced` are exact-wrapper judge-image screens.

Decision: if native probes/TASK and resource assumptions pass, nominate every
changed full-decode point below1 with paired SE and seed samples; P4's held-out
six-load G0 confirms the composition. If real occupancy is not restored, this
run has not tested that explanation. If the admitted form still loses, close
this score-remap form; direct accumulator softmax/PV mappings remain open.
A:A drift is reported without adding a per-load win gate.

Exact wrapper28483f42 passed488/488TASK checks,13host and2static checks
in both modes (121.39/285.45s private lab). All native probes remain required.
Sent as decode-host check **31730357bae64c9c**, priority3; `queue.json` keeps
full source/payload hashes and sender identity. Results pending;
`read_rebuild.py --reuse <receipt>` computes every paired seed score and point SE.
