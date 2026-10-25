# P5b full attention

Active kernel-round work: full decode16/32 and full verify16/32. Base S14 final
**f1c77ce6**; no variant from this lane is promoted. P5 owns sliding and decode1.
This directory is not parked serving work.

Two independent, screened batches are held for **2026-10-26 09:00 KST**
(epoch1792972800), on the proj host:

| Batch | Wrapper /sender PID | Priority | Working set |
|---|---|---:|---|
|Full verify softmax, eight arms|030864131ce8d08f /3171016|2.0|[softmax](softmax/README.md)|
|Full decode softmax, ten arms ×eight seeds|06e39759a67ca297 /949154|2.5|[decode](decode/README.md)|

The verify sender and its frozen bytes remain unchanged. Decode tests mixed
MUFU/FMA exponentials and FP16-P tensor-core sums directly in attn_main, plus
the native two-warpgroup q_len1 reference/splits/ping-pong. Every arm has local
TASK/cache/capture coverage and strict native proofs. The decode batch does not
depend on which verify arm wins; its README gives the integration decision rule.

No judge job IDs exist before acceptance. Sender logs are under runs/detached,
and the processes collect canonical receipts into runs/ after their sends.
Read both receipts, then screen and pair a supported winner in the whole layer
on identical inputs and fresh seeds. Component or sm120 timing does not promote.
