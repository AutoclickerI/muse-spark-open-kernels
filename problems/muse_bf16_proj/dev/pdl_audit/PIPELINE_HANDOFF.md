# P7 → P10b: preserve the weight pipeline when adding PDL

S14 audit base `f1c77ce6`; original/fixed disassembly and compiler resource
records are in this directory. Native direct GF and coded CD GEMVs already
preload initial weight vectors before their dependency wait. Their projection
bodies are retained by P7.

For a Triton GEMV, pulling one weight tile into a separate register value and
using `if kb == kb0` in the main loop is expensive here. Representative balanced
GEMV changes from 48→150 registers and 40→8 LDGSTS instructions plus 0→16 LDG;
the eight tensor-core instructions are retained. The broad recipe loses
11.5% in paired six-load graph speedup (`8db4213240c6420d`, sixteen seeds), with
8–44 us added at small M and nearly equal added time in sliding/full attention.

`build_hint.py` instead issues the first tile's weight cache requests before
wait and preserves the original software pipeline. It restores the baseline
register allocation and LDGSTS schedule. Direct attribution in `0f74021a2c53445e` repeats the register-tile loss
(F .8870 six-load speedup); cache hints reach 1.00269 and metadata-only 1.00317.
These small gains need shared-JIT confirmation: unchanged kernels originally
had separate device handles. The corrected payload is locally validated but
H200 quota-blocked until 09:00 KST. No projection recipe is promoted.

GLUE_CL_CUDA has a separate source-to-machine-code issue: 17/19 specializations
sink intended immutable norm/bias/local-start loads past ACQBULK. Inline vector
PTX restores their placement. Two C2/256 boundary variants already preload.
Triton glue's existing early vector loads generally remain early; inspect the
emitted code before replacing them. Native scale/position/RoPE preload repairs
are summarized in README.md. Never move activations, QKV or route outputs ahead
of their producing dependency.
