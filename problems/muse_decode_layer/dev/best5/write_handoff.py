"""Create the final handoff from the completed normal/cold forced screens."""
from pathlib import Path
import ast,hashlib,json,re
ROOT=Path(__file__).resolve().parents[4]
DEV=ROOT/'problems/muse_decode_layer/dev'
HERE=Path(__file__).parent
CAND=DEV.parent/'cand_r3_best5.py'


def main():
    sha=hashlib.sha256(CAND.read_bytes()).hexdigest()
    screens={}
    for name,path in [('normal',DEV/'best5_local.json'),('forced',DEV/'best5_forced_local.json')]:
        data=json.loads(path.read_text());assert data['sha256']==sha
        for field,count in [('tests',13),('shapes',12),('static',13)]:
            assert len(data[field])==count and all(x['ok'] for x in data[field]),(name,field)
        screens[name]=data
    log=(DEV/'best5_forced_cold.log').read_text()
    elapsed=float(re.search(r'^BEST_PREPARE elapsed=([\d.]+)s',log,re.M)[1])
    phases=ast.literal_eval(re.search(r'^BEST_PHASES seconds=(.*)$',log,re.M)[1])
    rows={int(m):ast.literal_eval(row) for m,row in re.findall(r'^BEST5_NT M=(\d+) (.*)$',log,re.M)}
    assert set(rows)=={32,128,256} and all(x['reason']=='measured' for x in rows.values())
    provenance=json.loads((HERE/'audit.json').read_text());assert provenance['source_sha256']==sha
    assert len(provenance['expired_deadline_tuners'])==6
    result=dict(sha256=sha,cold_prepare_s=elapsed,cold_forced_prepare_s=screens['forced']['prepare_s'],
                phases=phases,moe_pairs=rows,provenance=provenance,screens=screens)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    text=f'''# Decode best5 handoff

[cand_r3_best5.py](../../cand_r3_best5.py), SHA-256
`{sha}`. Not sent by this lane; Jihoon sends it as a decode check.

Cold candidate prepare: **{elapsed:.3f} s**; including the extra forced-glue checks:
**{screens['forced']['prepare_s']:.3f} s**. Extension, Triton and driver caches were empty and ccache disabled.
All M32/128/256 measurements completed. Cold phase seconds: `{phases}`.
Normal prepare was {screens['normal']['prepare_s']:.3f} s, including the first NT2 build in the existing local cache.
Normal and forced screens each pass **13 tests, 12 fresh-input graph shapes and 13 static checks**,
including compilation of every CUDA source for sm_90a.

Base is exact best `812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2`.
Its NT1 CUDA source and all non-MoE native sources are byte-identical. `kernel_fn`, projections,
attention choices/gates, routing, boundary kernels and PREFIX are unchanged. Only their prepare tuners
receive checks of one deadline, measured from prepare entry (240 s). The NT2 extension compiles alongside
initial weight preparation. It uses the byte-identical kernel body from accepted H200 lab
`3f2b8da0f1d244d7`; its bindings instantiate NT2 only. No CUTLASS engine, reuse engine, NT4 or new native
kernel is added. The NT1/NT2 extensions share the existing byte-preserving NVFP4/E4M3 weight bank.

For each M32/128/256, three fresh inputs reproduce the public stream/draft distribution with this run’s
actual loaded router, bias, norms and pooler. Both kernels must pass an independent FP32 expert oracle
before and after graph timing (relative L2 <=1%, worst row <=5%). Oracle dequantized weights are temporary
scratch and never enter returned state. Both captured graphs see the same changed inputs. Twelve paired
cold replays alternate launch order; selection uses the median geometric ratio across adjacent swapped
pairs. Other M keep exact best NT1. The three compact `BEST5_NT` lines are printed last in prepare.

| M | NT1 µs | NT2 µs | Paired NT2/NT1 | Selected | Maximum FP32 relative L2 |
|---:|---:|---:|---:|---:|---:|
'''
    for m,row in rows.items():
        text+=f'| {m} | {row["nt1_us"]:.3f} | {row["nt2_us"]:.3f} | {row["paired"]:.6f} | NT{row["selected"]} | {max(row["fp32"]):.6f} |\n'
    text+='''
These are local sm_120 choices, not H200 predictions. The exact H200-loaded router determines that run’s
choice. [Validation, routed-pair/expert/hot counts and all phase timings](validation.json).

Forced glue uses the existing best attention stand-ins, forces every attention family and fused PREFIX,
and then forces the real NT2 extension in complete layer graphs at all three eligible M. Both native
MoE variants execute locally; Hopper attention launches use portable kernels, while their actual native
sources compile during cold prepare. H200 best5 prepare, selections, agreement and speed remain unverified.

The [audit](audit.py) verifies native provenance and unchanged stage bodies. It also expires the shared
deadline and verifies all six tuners return without allocating GPU inputs or compiling an extension.
No standing rule was added; the omni abort remains the existing FINDINGS failure.

## Reproduce

```sh
python problems/muse_decode_layer/dev/best5/build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/best5/audit.py --deadline

CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \\
  problems/muse_decode_layer/cand_r3_best5.py --no-baseline \\
  --json problems/muse_decode_layer/dev/best5_local.json

# Use a new, empty directory prefix to repeat the cold measurement.
CUDA_VISIBLE_DEVICES=0 tools/img env CCACHE_DISABLE=1 \\
  TORCH_EXTENSIONS_DIR=/tmp/muse-best5-cold-01/extensions \\
  TRITON_CACHE_DIR=/tmp/muse-best5-cold-01/triton \\
  CUDA_CACHE_PATH=/tmp/muse-best5-cold-01/cuda \\
  TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \\
  python problems/muse_decode_layer/dev/best5/force.py --cold-native

python problems/muse_decode_layer/dev/best5/write_handoff.py
```
'''
    (HERE/'README.md').write_text(text)
    print(sha,elapsed,screens['forced']['prepare_s'])

if __name__=='__main__':main()
