Read before changing an expert-bank packer or reader: this is the geometry and scale-storage contract.
Serves: G2.

# The NVFP4 expert bank (canonical V6)

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

A recoded bank stores each E4M3 scale as a uint8 exponent/mantissa code; sign and zero fold into
FP4 codes. The reader reconstructs the exact value in the timed kernel. This is a lossless
encoding, distinct from storing a prepared BF16-valued word in int16. Global gate/up scales
stay independent. Duplicate layouts cost another 864 MiB/layer; no decoded numeric bank is
part of this contract. Admissibility: CONSTRAINTS 5; measured gains: FINDINGS "Weight encoding".
