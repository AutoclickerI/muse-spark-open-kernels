"""Write the current quarantine handoff from matching completed local screens."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parents[1]/'cand_r3_omni.py'


def main():
    sha=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    normal=json.loads((HERE/'quarantined_screen.json').read_text())
    forced=json.loads((HERE.parent/'omni_forced_local.json').read_text())
    for d in (normal,forced):
        assert d['sha256']==sha
        for kind in ('tests','shapes','static'):
            assert all(r['ok'] for r in d.get(kind,[]))
    manifest=dict(sha256=sha,mixed_moe='removed; native source is host-geometry-only experimental',
                  normal=normal,forced=forced,h200='not sent; no native mixed numeric pass')
    (HERE/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    text=f"""# Decode omni quarantine handoff

[cand_r3_omni.py](../../cand_r3_omni.py), SHA256 `{sha}`.
No candidate was sent by this lane. The owner screens/sends candidates and commits.

The CUTLASS mixed-input MoE engine has been removed from the candidate, including its
embedded CUDA source, build factory and dispatch. The MoE tuner retains V9, NT2 and NT4.
The prior a704a37c run aborted in host TMA descriptor construction, before a numeric gate
could reject it. Its previous local portable launch checks did not exercise that setup.

The repair in [mixed.cu](mixed.cu) remains experimental. Host setup now uses a real fixed
capacity problem shape, canonical nonzero strides and a 16-byte contiguous E4M3 scale
atom. Original all-ones dummy dimensions and the eight-byte scale atom were independent
TMA errors. GPU-stride arrays are never dereferenced by host can_implement. The raw scales
remain E4M3; no prepared expanded weight bank is made. The stock CUTLASS group-K64 scale
broadcast cannot implement group16; this source is a private group16 specialization.

[Host reproducer](../../../muse_moe_w4a16/dev/mixed_wgmma/check_cutlass_host.py)
builds the original and repaired sm90a code and calls real TMA setup on local sm120.
This is a host geometry check; mixed GEMM agreement and performance are unverified.
The engine stays out of candidates until a clean numeric H200 lab.

Current normal screen: {len(normal['tests'])} official tests, {len(normal['static'])} static
checks, prepare {normal['prepare_s']:.2f}s. Forced projection/attention glue screen:
{len(forced['tests'])} tests, {len(forced['shapes'])} graph shapes and {len(forced['static'])}
static checks, prepare {forced['prepare_s']:.2f}s. [Exact evidence](validation.json).
NT2/NT4 execute natively; Hopper attention and projection launches use explicit portable
stand-ins. These local timings do not rank H200 kernels. Previous cold prepare timings
belonged to the withdrawn implementation and do not describe this candidate.

Reproduce with tools/img python tools/local_eval.py problems/muse_decode_layer/cand_r3_omni.py
--tests-only, and tools/img python problems/muse_decode_layer/dev/omni/force.py.
"""
    (HERE/'README.md').write_text(text)
    print(sha)


if __name__=='__main__':main()
