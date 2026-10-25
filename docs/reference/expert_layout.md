# The NVFP4 expert bank (canonical V6)
Serves: G2. Read for the expert bank's geometry and scale storage: the contract the decode candidates' expert
readers cite ("bank contract docs/reference/expert_layout.md").

## Canonical V6 tiles

`prepare()` repacks each layer's 32 local experts into tiles; only the physical order changes, codes stay packed
E2M1 and scales stay E4M3 bytes. W13 pairs eight gate rows with the matching eight up rows in each 16-row tile; W2
uses natural 16-row tiles. Each tile holds 32,768 code bytes and 4,096 scale bytes (36,864 B):

| Tensor | Shape per layer | Bytes | MiB |
|---|---:|---:|---:|
| W13 tile bank | `[32, 512, 36864]` | 603,979,776 | 576 |
| W2 tile bank | `[32, 256, 36864]` | 301,989,888 | 288 |
| One bank | | 905,969,664 | 864 |

That is 9/16 byte per weight (packed E2M1 plus one E4M3 byte per 16 weights). The FP32 global scales `gs13_gate`,
`gs13_up` and `gs2` stay separate. The raw reader converts E4M3 scale bytes in the kernel.

## Recoded scale banks

From S1 on, candidates also build a recoded bank in `prepare()`: each E4M3 scale is re-encoded as a uint8 exponent
and mantissa, with its sign and zero folded into the FP4 codes. The recode is exact (a lossless re-encoding,
CONSTRAINTS "Boundary cases"), and it removes per-call scale conversion: NT1 runs 9-12% faster at M=8-128
(`58ae2e2c`), and the fused M=256 path 5.4% (`9e8a6a15`). S1 reads it for NT1 at M=1-128 and keeps raw storage for its
other paths; S3 and S6 read recoded W13 and W2 at M=256 too. A second complete layout costs another 864 MiB per
layer, which the one-layer kernel problems have room for; a dequantized (BF16/FP8) copy is forbidden (CONSTRAINTS
"Boundary cases", withdrawn `07d07d96`). Readers that convert raw scales per call should be measured against a
recoded variant.

## Serving loader (parked until the serving round, R-20261025-1)

Ports and gates: [serving/moe_port.md](../serving/moe_port.md), [serving/serving_levers.md](../serving/serving_levers.md).

- The serving loader keeps **one stock Marlin packed bank** for every M (`_muse_v4_state=None`) and releases
  serialized aliases after Marlin's repack; no dequantized copy is kept.
- A V9-small / Marlin-large hybrid has a shared-bank reader awaiting H200 admission: `muse_spark_expert_bank.py`
  joins stock Marlin's code and scale arrays into one allocation per W13/W2 (the original tensors stay
  byte-identical views); `muse_marlin_v9_reader.py` reads them directly and inverts Marlin's scale-byte transform in
  the kernel. `MARLIN_READER_ADMITTED=False` keeps stock until the reader's own timing wins. Intended policy: V9 at
  M <= 256, Marlin above (including 257-511).
- **Global-scale convention:** stock Marlin shares its BF16-rounded gate global between both W13 halves. Comparing
  against the original independent FP32 globals gave a 0.021484375 mismatch; matching stock's convention restores
  bit equality between native V9 and the direct reader.
- `muse_spark_serving.run_v9_dense` caps calls at 256; grouped and Humming are never selected in serving
  (real-weight H200 evidence rejects Humming/V11 agreement; grouped loses under skew).
- Memory per EP8 rank: 60 routed layers x 32 local experts = 96,636,764,160 parameters = 54.358 GB (50.625 GiB) at
  9/16 B. A second complete layout would cost another 54.358 GB, leaving 11.13 GB of a 119.85 GB static budget before
  dense weights, drafter, graph pools and KV; a per-call one-layer repack was rejected (377 us per layer modeled
  traffic at c32 vs a 114 us break-even).
- Verification: `test_shared_expert_bank.py` (13 shapes M=1-2048; M <= 256 relative L2 <= 0.00538 against stock,
  Marlin path bit-exact through 753) and `test_marlin_v9_reader.py` (all 127 positive finite E4M3 encodings,
  native-V9 bit parity, eager and graph replays). H200 admission lab source `6a3168ef5d749e10`. Local sm_120 and
  sm_90a compiles don't establish H200 cooperative execution, TP8/EP8 FA3 agreement or full-checkpoint residency.
