# S15 experimental prepare-tensor audit

Candidate **60254883** on exact S14 **f1c77ce6**. Combined normal and forced local_eval each pass
13 TASK tests,12 graph shapes,25 static checks. Recursive retained-state inventories contain92/132 tensors.
Projection-only **e3ce8f15**, last-only **6a455ff6** and regmax-only **d30ce9f9** independently pass both modes.

| Retained data | Encoding / precision |
|---|---|
| S14 expert banks | Original E2M1 codes plus E4M3 scale bytes; lossless PERMW/light layouts,9 bytes per16 weights; no decoded expert weights/scales retained |
| S14 coded_dense | Original lossless BF1612-bit coding, exact exponent metadata and original-bit escapes |
| NF small-M wp | uint8 lossless12-bit BF16 sign/mantissa/exponent-window codes, tiled/stream-permuted |
| NF esc / row tables | int32 coordinates, bit-exact BF16 escape correction, exact high-byte table bits; no widened per-weight values |
| NF O32 raw layout | Original BF16 values in a bijective XOR swizzle/transpose; independent inverse checks every BF16 bit at its coordinate |
| NF roles | Original BF16 dense matrices/slices; unused fused pool/router holds BF16 hi/lo halves of the permitted FP32 router |
| Attention partials/counters | Runtime FP32 partials and int32 arrival counters; all contributions processed, no cached input-derived results |

`audit.py` traverses dictionaries, adapter objects and closures, verifies original NF BF16 roles, expert
bank formats and original coded_dense bit reconstruction. Both inventories retain only BF16/uint8/int32
NF GPU tensors locally; Hopper-only plans are materialized independently in `audit_layouts.py`.
All9 selected NF layouts pass. The exact selected packers inverse-decode all BF16 bits before returning;
the final stream-code payload/escape/table dtypes are audited after cloning codes and source weights.
The MW unpool32 packer asserts pack_exact. `hopper_layout_inventory.json` records every selected row.
These temporary audit banks are not added to production state. Pinned `proj/native*_compile.json`
records the unchanged port's16 adapted and16 original sm90a compilations; code, vendor and reports are
hashed in dependencies.json. New attention CUDA also compiles through the25-check combined audit.

Normal/forced screens explicitly identify local launch stand-ins: Hopper N128 verify uses the admitted
S14 N64 launch; last-CTA uses its actual Python allocation/counter/partial ABI with direct-partials
arithmetic plus a native pointer-copy/reset launch. Native NF cluster/WGMMA rows use the port's declared
local exact-format arithmetic stand-ins. No stand-in code is shipped to H200. H200 native admission,
including official changed-full correctness, remains pending. New regmax prepare asserts that both rows
are installed; it cannot silently pass a rejected new attention path.

The complete wrapper is losslessly packed to634,595 bytes; its plaintext MIT copyright/permission notice
remains intact. Expanded candidate source is audited separately (25 static checks); bootstrap/wrapper
static checks also pass. Normal/forced wrapper runs preserve all12 shapes,16 seeds,32 rounds at T<=8,
8 elsewhere and the full1GiB read flush. Only the local wall deadline is extended to accommodate sm120;
the shipped285-second prepare deadline remains unchanged. All seven off arms share35 identical Triton
objects with the host; changed VQ8 source keeps a private native module cache and the original off-arm
module/setting. Full S14 projection/MoE plans, banks and counters are shared unchanged.
