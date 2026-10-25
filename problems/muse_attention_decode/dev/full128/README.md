# full128

Register-fold N128 full-verify stage4 handoff; changed-full host correctness is queued.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [build.py](build.py): Native N128 cached tiles, two KV heads per stream-K CTA, N64 virtual tile and unchanged combine.
- [check_tile.py](check_tile.py): Validate the exact N128 QK/softmax/RS-PV helpers with FP8 data on the portable GPU.
- [full128.cu](full128.cu): Device source or payload template used by retained code.
- [patch_full128.py](patch_full128.py): Replace only frozen S11's full-verify native vq8 cell; paired H200 evidence must select the row settings.
- [s11_regmax_full_verify.patch](s11_regmax_full_verify.patch): Pinned integration patch.
- [smoke.py](smoke.py): Exercise the exact embedded layer/timeline payload on the portable GPU before H200.
- [tile.cu.inc](tile.cu.inc): Device source or payload template used by retained code.

Handoff: [s11_regmax_full_verify.patch](s11_regmax_full_verify.patch), SHA256 `ca730e6102aa245709f908f56fe7d4df2d527e51b79ad54e9e8ae45e78ed2a76`, applies onto S11 acda6c62.

Pending H200 senders (not judge-accepted at handoff):

- PID 2411468, wrapper `0ccc25484dbe5810`, [lab_p5_global_attention.py](../../../muse_decode_layer/lab/lab_p5_global_attention.py); [sender log](../../../../runs/detached/0ccc25484dbe5810.log). Reset 2026-10-26 09:00/09:01 KST; dispatch metadata: `runs/waiting/2411468.json`.

Next: Collect global-attention wrapper 0ccc25484dbe5810 for official changed-full correctness; select future work from paired layer critical increments. Evidence: 9d9a01e1f6db4d79.
