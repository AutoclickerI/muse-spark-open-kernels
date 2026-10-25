# full head

Dependency working set retained for `problems/muse_attention_decode/dev/cluster_sliding/build_attention_board.py`.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [build.py](build.py): N128 full attention, one KV head per 192-thread CTA; two resident CTAs/SM at stage4.
- [driver.py](driver.py): Driver for vq8 (q_len 8 attention, one KV head per CTA, stream-K, native FP8 WGMMA on sm_90a).
- [full_head.cu](full_head.cu): Device source or payload template used by retained code.

Handoff: No independent integration handoff..

Pending H200 senders: none owned here; follow the consuming lane.

Next: Resume from the consuming lane; this directory has no separate pending experiment or promoted patch.
