# P10 next decision after the public-port whole graph

The native S19 stream scenario is concentrated at pool/router16 (11.580us of
d16's13.278), then QKV32 (3.000), pool/router32 (2.848), and unpooler32 (2.536)
of d32's8.393. Smaller carriers: unpool16 .918, O16 .478, QKV16 .302us. Gate has
zero positive exposed surplus; O32 .008758us is too small for a separate build
and keeps the parent port choice. [SIZING.md](SIZING.md) and `sizing.json` derive
every site from native `e70024b150b942fa`, cross-checked against the independent
S19 timeline `9c16c850f7b941a5`. T32 pool/router is currently **dense BF16, S6**;
the phrase “coded gap” does not apply to that bank. R1.032884 is a model using
S19's actual selected bytes, not a physical or universal ceiling. New coding can
also lower those bytes. PDL residence is never counted as independently saved.

The candidate uses the already licensed/reviewed NF exact12 payload and escape
patcher, with more smaller CTAs and DSM reduction. It avoids the rejected Triton
decoded-W staging path and its255-register/1016-byte stack. Every result is FP32
S1 behind the existing layer projection ABI; the normal neighbours run unchanged.

| Leave-out group | Candidate |
|---|---|
|Q16|NF12 S4/WK4/ST1, DSM RED3, first codewords before activation wait; QKV3 S and QKV/gate F|
|Q32|Same schedule at MT4 for QKV/gate S and QKV3 F|
|O16|NF12 S1/WK4/ST2, first two codeword units before activation wait|
|P16|NF12 S4/WK4/ST1 DSM RED3|
|P32|Tiled raw TMA BN64/BK128/ST2/S3, DSM FP32 reduction; explicit PDL launch|
|U16/U32|NF12 S2/WK4/ST1 DSM RED3, full8192-row output|

`CODED_P32` additionally tests NF12 S4/WK4/ST1 at T32. It shares P16's stream bank
and adds only a template specialization. The raw and coded alternatives pay their
actual whole-layer work. Native bank bytes, prepare seconds per bank, compile and
warm costs are recorded; selected prepare and H200 gains remain unmeasured.

The TMA first tile is peeled. A scalar from that weight tile feeds an opaque
post-`griddepcontrol.wait` zero used in **every activation descriptor coordinate**.
This prevents software pipelining from reading activations early. sm90 PTX proves
one weight descriptor before the wait and the activation descriptor after it.
The launcher also explicitly has `launch_pdl=True`. A source-only wait inside the
loop was insufficient: the compiler hoisted both descriptors ahead of it.

That same defect existed in parent sender3347492's two early-TMA schedules.
`parent_repair.json` preserves the unsubmitted old snapshot/manifest and the
replacement identity. It was superseded through the normal sender before quota
opened; **no quota used, same arms/seeds/rounds/checkers**. The corrected parent is
sender3966799 (`5a8ae27c764a6f9b`), still waiting for10-27 09:01KST.

`dispatch.py --start` creates one dependency dispatcher. It waits on that exact
sender's pidfd completion event, validates the exact native receipt, runs
`../whole_read.py`, drops in-graph mean-negative points, and hands the remaining
points/pool schedules to P4 as an exact `portfolio.py` patch. Uncertain positive
means retain their CIs as exploratory nominees; no stage/load veto replaces G0.
After completion it reads the immutable canonical receipt source; the sender
removes its temporary `runs/sent` snapshot. It then substitutes only constrained parent-selection constants into the locally
admitted next batch and calls the normal `h200_lab` sender with `--lane P10`.
PLAN rank controls priority. No polling, sleep cadence, direct service API, signal
to another process, repeated advice, background git writes, or duplicate send.
The dispatcher exits after the next send. Invalid/incomplete parent evidence
stops it with an explicit failure instead of inventing a portfolio.

The next batch freezes S21eacc087b from an immutable git blob, plus the measured
parent portfolio. INC=S21, A/AA=that portfolio, FULL=all seven new groups, seven
leave-outs, and the coded P32 alternative. Eight new input seeds, fresh weights
and router. Every arm gets published layer TASK; new kernels also get original
projection/router random and one-column checks. Timings use all cyclic positions
and their exact reverses:16 rounds at T16/T32,8 elsewhere; every round retained.
Untouched leave-out shapes have exactly the FULL graph and share its samples.
This is selection evidence. P4 still confirms the chosen composition with its
held-out paired six-load G0, including nearby checks and full prepare.

Local admission:512layer+208strict projection/router TASK for both default and
parent-pool stress portfolios; exact inverse stream/escape/origin checks on all
five distinct packed banks; full packet plus2k suffix guard. Nine NF kernels
cross-compile sm90a at64–96regs with zero spills. The TMA cluster cross-compiles
with explicit PDL, valid shared-memory admission, and guarded activation order.
`candidate_validation.json` records official normal/forced13TASK/12graphs/6static
screens. Local body arithmetic stand-ins establish glue and encoding coverage,
not native speed. `admission.json` pins all code while allowing only validated
parent point/pool constants to change after the receipt.

After the next native receipt: `python3 problems/muse_bf16_proj/dev/public_ports/night/read.py RECEIPT`. It reports
INC/FULL, A/FULL, every NO_site/FULL, and coded-vs-raw P32 with Student-t95% CIs
using the larger primary/A:A log SE. All native rounds stay in `results.json`.
No new G0 winner or native night result exists yet.


Queue state: `queue.json`, dispatcher4088520, dependency3966799. It is currently
blocked on the actual parent-completion event, not a timer. Host Python and glibc
lack a named pidfd wrapper; the worker uses Linux x86_64 syscall434. Actual child
exit and filesystem-close tests pass in `pipeline_validation.json`. Failed
startup attempts sent nothing and are retained in `startup_failure.json`.
