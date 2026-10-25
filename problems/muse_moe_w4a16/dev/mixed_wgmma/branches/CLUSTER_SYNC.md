# Multicast ring synchronization audit

Audited CUDA: **075ce977** (`cluster_manifest.json`). This is an instruction-contract argument,
checked by `cluster_sync_check.py`; it is not an exhaustive hardware race test. H200 job
**5fdb668cc2964cdf** completed both CS1 and CS4 on96 fixtures, all output gates passing.

The [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) defines the contracts
used here: multicast's mask identifies destination CTA ranks; its completion credits their barriers;
cluster synchronization orders participating CTAs; `wgmma.wait_group N` retires older groups while
allowing at most the newest N to remain pending. TMA and WGMMA use the async proxy, while LDGSTS uses
the generic proxy ([CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)).
The analysis below is derived from those contracts and this kernel's address/event schedule.

## Participants, addresses and publication

Launch dimensions are `(CS,1,1)`, with grid.x divisible byCS. Logical CTA rank is `blockIdx.x % CS`.
For every cluster unit, all ranks compute the same expert, count, token-row group, native N and stage
count. Only their channel tile differs. Every conditional around cluster barriers is therefore
uniform within the cluster, including empty experts and the257-token tail.

Each CTA initializes its own mbarriers once, publishes initialization with
`fence.mbarrier_init.release.cluster`, then joins the initial cluster barrier. All participating
ranks have identical shared-memory layouts and barrier offsets. Before each multicast, thread0 in
every destination arms its own barrier with one arrival and expected bytes `NT * 128 * 2`.
The cluster barrier occurs after all arms and before the leader's two TMA instructions.

Only thread0 of rank0 issues those two copies. Each moves `NT * 64 * 2` bytes to every rank selected
by mask `(1<<CS)-1`, at identical CTA-relative destination and barrier offsets. Their sum is exactly
the expected `NT * 128 * 2` bytes **per destination**; no factorCS belongs in a local byte count.
Every destination waits for its own completion before consuming its X stage.

The packed Q and scale copies have an independent dependency: `cp.async.wait_group 0` followed by
`__syncthreads` before LDS/dequant. TMA completion alone is not used to order them. Their storage ends
at `S * (4096+512)`; the X base rounds this up to1024B. Thus generic-proxy Q/scale accesses and
async-proxy X accesses do not alias. Hopper's X is written by TMA and read by WGMMA; no generic X
stores intervene. The parallel-copy diagnostic instead includes the generic-to-async publication
fence because its X writer is LDGSTS.

## Slot and register reuse

Number WGMMA groups by `g(k,sub) = 2*k + sub`, with four K16 MMAs per group. A stage contains two such
groups. All128 threads execute the same fence/commit/wait sequence.

For **S2**, the next stage is issued just after sub0 of stepk commits and waits with limit1. This retires
all groups through `2*k-1`. The slot being overwritten is `(k+1)%2`, last used at stepk-1; its last reader
is exactly group `2*(k-1)+1 = 2*k-1`. Every CTA performs that wait before the cluster barrier, so even
the slowest rank has retired its old reads before the leader overwrites the slot. The newest group
`2*k` can remain pending because it reads the other slot.

For **S1**, both halves use the only slot. The explicit `wait_group 0` after sub1 retires every read
before the next arm/cluster barrier/copy sequence overwrites it.

The two A-register buffers follow the same group order. Before writing sub0's buffer at stepk, the
preceding step's sub1 wait has retired through `2*k-2`, including that buffer's previous group.
Before writing sub1's buffer, this step's sub0 wait retires through `2*k-1`, including its previous
use. `keep` reads the opposite, already-retired buffer. Generic accumulator/epilogue accesses occur
only after `wait_group 0`. SASS preserves group-limit1; the audited kernels have no register spills.

Each barrier is reused32/S times per complete K4096 tile. This is16 for S2 and32 for S1, both even,
so each new expert/channel tile starts again at parity0 without reinitialization. A final wait0 and
cluster barrier drain readers before any rank advances to another tile. At kernel exit all ranks
join once more before local barrier invalidation; no multicast remains in flight.

## Checked limits and evidence

`cluster_sync.json` records **36,864 ordering/byte-count checks** and **1,888 launch-class checks**.
The model keeps every group pending as long as its wait allows, checks all width classes and both
CS1/CS4, and changes widths across tasks. Negative controls expose three hazards: limit2 permits
A-buffer reuse too early; omitting all-rank retirement permits overwriting a peer's live slot;
an odd number of single-stage K steps breaks the task-to-task parity reset.

`cluster_validation.json` additionally records392 poisoned captured checks on sm120, exercising real
portable math, gather, class boundaries, nearby sizes and stream joins. It forcesCS4 toCS1 and does
not exercise multicast hardware. H200's96 passing fixtures cover actual multicast execution; they
do not prove absence of every possible hardware interleaving. The ordering argument above is the
reason slot reuse is safe under the stated PTX contracts.
