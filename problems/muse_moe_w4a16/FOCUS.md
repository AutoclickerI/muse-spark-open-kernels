# MoE focus
Serves: G1, G2.

G1/G2: the decode-layer score (R-20261025-1). Own M128 MoE.
Bar: NT1-recoded 283.6 us on paired H200 routes (58ae2e2c); floor ~106 us.
Free lane owns M256/W2, whose distributed finalizer is reused in this lane.
Current lab accepted on MoE: send a1762556e26931ef, judge dc3fea0a20a349bf.
Automatically batched with the small-M instruction-limit lab, priority 0.4.
Detached log inspected. Payload: prof/lab/moe_m128_issue_limit.py.
Sources, SASS counts, hashes and model: dev/mixed_wgmma/m128/issue_limit/.

Measured sm90a SASS: NT1 unrolled K512 is 492 -> 476 instructions/256 weights
with a single LOP3 blend. F128 adaptive N32 W13 is 84.75 -> 81.25 instructions
per16 weights; address work is37/84.75. Smaller D32/D16 loops use65/59.
At the supplied four-warp issue budget, canonical-byte estimates are NT1
9.99 TB/s before rereads; F128 N32 3.62 -> 3.78 TB/s. Static body counts include
conditional paths; these are model estimates, not measured H200 ceilings.
Real-route tile decode rereads NT8/16/32/128 =2.176/1.478/1.157/1.000.
NT1's logical972.8 MB reads cost243.2 us at4 TB/s without L2 reuse, versus
97.4 us ideal aggregate issue. NT1's binding hardware limit remains unresolved.

The lab pairs actual NT1-recoded, one-LOP3 NT1, F128 recode/word scales and
D16/D32 word scales with distributed W2 against identical real routes.
Twelve weight/router seeds x3 fixtures, six cold128MiB orders, A:A noise;
separate W13/W2 phase graphs and NT1 per-CTA globaltimer distributions.
Local252 graph checks +42 signed/zero/subnormal/route-transition gates pass;
4064 scale/code values exact;17 native entries no spills/C7513,16 safe K64 paths.
Wrapper5 tests/6 scored graph shapes/6 static checks pass; full lab26.4s/78s.
Prior471f6d6216804dda had no receipt at this turn's read and has no phase timers.
Next resume: read dc3fea0a20a349bf once. Promote only a complete paired H200
winner; locally gated NT1lop drop-in and SHA are in issue_limit/dropin_manifest.json.
