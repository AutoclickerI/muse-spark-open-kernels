"""Write the final validation handoff from completed local screens."""
from pathlib import Path
import ast,hashlib,json,re
ROOT=Path(__file__).resolve().parents[4]
DEV=ROOT/'problems/muse_decode_layer/dev'
CANDIDATE=DEV.parent/'cand_r3_omni.py'

def screen(path,sha):
    d=json.loads(path.read_text());assert d['sha256']==sha,(path,d['sha256'],sha)
    for kind in ('tests','shapes','static'):assert all(r['ok'] for r in d[kind]),(path,kind)
    return {k:d[k] for k in ('sha256','prepare_s','gpu','cc','tests','shapes','static')}

def main():
    sha=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    normal=screen(DEV/'omni_local.json',sha);forced=screen(DEV/'omni_forced_cold_local.json',sha)
    log=(DEV/'omni_forced_cold_final2.log').read_text()
    elapsed=float(re.search(r'OMNI_PREPARE seconds=([\d.]+)',log)[1])
    compile_s=float(re.search(r'OMNI_COMPILE .* seconds ([\d.]+)',log)[1])
    rows=[]
    for line in log.splitlines():
        if not line.startswith('OMNI_SHAPE '):continue
        head,rest=line.split(' A ',1);_,b,l,w=head.split()
        attention,rest=rest.split(' P ',1);projection,rest=rest.split(' M ',1)
        moe,rest=rest.split(' G ',1);glue,pick=rest.split(' PICK ',1)
        rows.append(dict(batch=int(b),q_len=int(l),window=int(w),attention=ast.literal_eval(attention),
            projection=ast.literal_eval(projection),moe=ast.literal_eval(moe),glue=ast.literal_eval(glue),selection=pick))
    assert len(rows)==12
    def budget(d):
        return any(budget(v) if isinstance(v,dict) else v in ('B','skip') if isinstance(v,str) else False for v in d.values())
    assert not any(budget(row[k]) for row in rows for k in ('attention','projection','moe','glue'))
    command='''CUDA_VISIBLE_DEVICES=0 tools/img env CCACHE_DISABLE=1 \\
  TORCH_EXTENSIONS_DIR=/tmp/muse-omni-cold-final2/extensions \\
  TRITON_CACHE_DIR=/tmp/muse-omni-cold-final2/triton \\
  CUDA_CACHE_PATH=/tmp/muse-omni-cold-final2/cuda \\
  TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \\
  python problems/muse_decode_layer/dev/omni/force.py --cold-native'''
    artifact=dict(sha256=sha,cold_prepare_s=elapsed,cold_with_forced_checks_s=forced['prepare_s'],compile_s=compile_s,
        normal=normal,forced=forced,shape_tuners=rows,cold_command=command)
    (DEV/'omni/validation.json').write_text(json.dumps(artifact,indent=2)+'\n')
    text=f'''# Decode omni handoff

Candidate: [cand_r3_omni.py](../../cand_r3_omni.py). Not sent by this lane; Jihoon sends.
SHA-256: `{sha}`.

Cold candidate prepare is **{elapsed:.3f} s**, including **{compile_s:.3f} s** for 11 concurrent extension builds.
Including all extra forced-family FP32 checks: **{forced['prepare_s']:.3f} s**.
Extension, Triton and driver caches were empty; ccache was disabled. No recipe was skipped for budget.
Normal prepare with the existing local cache is {normal['prepare_s']:.3f} s.
Normal and forced-Hopper screens each pass **13 tests, 12 fresh-input graph shapes and 15 static checks**.
The static checks compile every CUDA literal for sm_90a.
[Machine-readable validation and every shape’s alternative times](validation.json).

The base is best `812f5f36`, preserving raw packed NVFP4/E4M3 V9 and its small pool/router GEMV.
MoE M128/256 compares V9 with NT2, NT4 and one CUTLASS mixed-input grouped recipe on three loaded-weight,
real-router fixtures, each checked against an independent FP32 oracle before and after paired timing.
Other M retain V9, matching the reuse drop-in’s scope. No prepared dequantized weight bank is retained.

The supplied mixed-input source was a compile scaffold. Omni adds a per-state grouped dispatch, GPU route
packing/shapes, correct device stride arrays and workspace reset on every timed replay. Its private
CUTLASS collective specializes group16 scaling with four scale columns per K64 tile, and widens E4M3
scales before multiplication. This avoids the stock collective’s K64 scale broadcast and FP8 product
rounding. [Readable native source](mixed.cu) exactly matches the embedded literal. NVCC `-O1` controls host
optimization; device optimization remains enabled. Tensor-only headers avoid pybind work in the CUDA TU.

Projection choices keep the measured cost-v9 Lt recipes and add two CUTLASS recipes per role at M128/256.
Only tile IDs 0, 2 and 6 are compiled. Recipe layout is `(tile, decomposition, splits, SMs, raster, swizzle, atomic)`:

| Role | C0 | C1 |
|---|---|---|
| qkv / unpooler | `(0,2,4,132,0,1,0)` | `(6,3,1,132,0,1,0)` |
| O | `(2,1,1,132,0,1,0)` | `(6,3,1,132,0,1,0)` |
| pool / router | `(0,2,8,132,0,1,0)` | `(2,3,1,132,0,1,0)` |

Attention includes the four-stage q8 `port_verify` from the current attention drop-in, all 24 XQA settings,
q1 attn_main and shortlists of FA3 recipes from cost-v9 and sweep `59e53e746df341a9`. The sweep receipt
again rejected every variant, including FA3: its timings inform the shortlist, while Omni’s own post-store
FP32 gate determines admission. Attention’s dynamic shared memory is 73,984 B for q1 and 104,448 B for q8;
PREFIX keeps best’s TB<=32 configuration (maximum 42,496 B).

Every new choice uses alternating paired cold graph timing, refreshed through three inputs. The shared
230-second deadline guards each tuner/recipe. The final 12 summary lines total 8,647 bytes and name every
alternative time or rejection. A labels: `base`, `main`, `verify`; `F<split><n/t/f>` is FA3 pack=None/True/False;
`W<stage>/<warps>`, `G<stage>/<warps>` and `P<stage>/<warps>` are XQA warp/group/FP8-PV.
P labels C0/C1 follow the table, L0/L1 are discovered Lt algorithms, NN/TN are the measured cost recipes.

Hopper-only attention and CUTLASS launches use portable stand-ins locally; all native extensions actually
compile during the cold prepare. The forced test executes GPU route planning, global scales, SwiGLU,
RMSNorm and combine with packed weights dequantized inside the stand-in GEMM. NT2/NT4 execute natively.
Extra preflight calls: mixed 48, CUTLASS 148, NT2 24, NT4 24, attention main 52 and XQA 624. Full layer graphs
force NT4 at M16/32, NT2 at M128, mixed at M256, CUTLASS projections, port_verify/XQA and fused PREFIX.
Native Hopper numerical agreement, selections, timing and cold prepare remain for the owner’s receipt.

## Reproduce

```sh
{command}

CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \\
  problems/muse_decode_layer/cand_r3_omni.py --no-baseline \\
  --json problems/muse_decode_layer/dev/omni_local.json

python problems/muse_decode_layer/dev/omni/lever_table.py
python problems/muse_decode_layer/dev/omni/write_handoff.py
```

## Forced tuner selections before the full-layer overrides

These are sm_120 diagnostics with attention/CUTLASS launch stand-ins; they do not rank H200 kernels.
Projection order is QKV, O, pool/router, unpooler. M<=8 uses the sliced unpooler.

| B | Q | Context | Attention; projections; MoE; PREFIX |
|---:|---:|---|---|
'''
    for row in rows:
        text+=f'| {row["batch"]} | {row["q_len"]} | {"sliding" if row["window"] else "full"} | `{row["selection"]}` |\n'
    (DEV/'omni/README.md').write_text(text)
    print(sha,elapsed,forced['prepare_s'],normal['prepare_s'])

if __name__=='__main__':main()
