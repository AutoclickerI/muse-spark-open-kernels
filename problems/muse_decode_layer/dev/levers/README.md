# Incumbent lever refresh

The committed `current.json` and `../NEXT_CHECKS.md` use S11 final `acda6c62` and job
`863f502c62e54d7b`. S12 revision `a92b6619`, job `937801feb77b4302`, is supported explicitly.
The builder checks both the immutable lab wrapper and its complete embedded timed source.
No H200 jobs were launched for this refresh; no candidate was changed or promoted.

Run from the repository root:

```bash
python3 problems/muse_decode_layer/dev/levers/build.py
python3 problems/muse_decode_layer/dev/levers/check.py
python3 problems/muse_decode_layer/dev/levers/scenarios.py
python3 problems/muse_decode_layer/dev/levers/build.py \
  --receipt problems/muse_decode_layer/lab/profiles/lab_decode_s12ab.937801feb77b4302.json \
  --out /tmp/muse_s12_levers.md
```

`--out` writes the JSON beside that Markdown unless `--json-out` is set, so checking an experiment
need not overwrite the incumbent model. Both source and profile-log hashes are in the JSON.
Historical scale/M256 corrections remain source-specific; S11/S12 apply neither. `scenarios.py`
recognizes both tags, applies no pending wins, and uses the same measured stream model.

The ideal floor still comes from `sim/bound.py` work at 4.8 TB/s and 989/1979 TF/s.
The empirical cold-read roof is separate: 2.36 us + bytes / 4.58 TB/s (`7b6c96b5`, FINDINGS).
S12's gate split bypasses coded qkv: its one visible coded GEMV is pool_router. Only its floor
shrinks, using the 12-bit payload and compulsory scale/offset metadata. Escape bytes are unknown
in the truncated log, so that coded footprint is explicitly a lower bound.

## Result and evidence

S11's six adjusted load sums (us), decode 1/16/32 then verify 1/16/32:
5936.117368, 11094.568321, 15076.832619, 8625.874825, 20624.928715, 28613.678907.
Their baseline/load ratios reproduce **1.7067674232784844x**. S12 reproduces
**1.7148028073027852x**. Cross-job host scores do not establish a paired win.

At 1.3x ideal floor, S11's lane headroom ranking is small-M MoE **11.70%**, attention **11.35%**,
glue **9.43%**, projections **7.67%**, MoE M256 **4.07%**, MoE M128 **3.16%**.
The full 1.5/1.3/1.15 tables, per-shape ratios and attribution ranges are in `../NEXT_CHECKS.md`.

The three stages with most absolute excess across six weighted load steps are MoE **15578.7 us**,
attention **8419.8 us**, boundaries x2 **5387.1 us**. The three largest individual shape/stage
excesses per layer are verify32 sliding MoE **107.00 us**, verify32 full attention **104.62 us**,
verify16 full attention **78.66 us**. These use visible rows and the existing N_full overlap
attribution, not recovered critical-path times. Decode16 full's O duration is 58 us, and its signed
layer-minus-profile residual is -75.89 us: that row exposes why PDL durations must remain caveated.

Existing paired evidence, `LOAD` lines in each profile's sibling `.log` (r > 1 means removal loses):

- S11 job `863f502c62e54d7b`: S10-back ratios **1.0488 / 1.0284 / 1.0423 / 1.0189 / 1.0579 / 1.0995**;
  vq8-off **1.0440 / 1.0821** at verify16/32; gate-off **1.0136 / 1.0010 / 1.0014** at decode1/16/32.
- S12 job `937801feb77b4302`: coded-off **1.0389** at decode1 (A:A **1.0108**); F4-off
  **1.0060 / 1.0073**; wide-off **0.9996 / 1.0015** at verify16/32 (noise).
  S11-back **1.0396 / 1.0046 / 1.0023 / 0.9954 / 1.0065 / 1.0085**. Verify1's small regression
  and cross-job profile differences are not a new promotion decision in this lane.

Eight offline tests pass: independent load-sum score reproduction, incumbent selection, rejection
of wrong wrapper/base SHAs, gate/QKV3 attribution, vq8 and LI_swap/F4 MoE accounting, floor/stream
separation, target monotonicity and independent scenario agreement, and prior S10 accounting.
Both builder CLIs and both scenario runs pass. Upstream sync changed no official task/rule files;
quota JSON differences already present in the shared tree were left out of this lane's commit.

## Integration handoff

This change applies to the lever analysis only, alongside S11 final and S12 a92b6619; it does not
patch either kernel. Use the lane's commit or its exported patch. Integration needs no GPU run.
Next: rerun the builder against the next promoted incumbent's source-verified profile. For kernel
work, use the updated ranking while resolving PDL exposure through paired stage/whole-layer labs.
