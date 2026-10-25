# vq8

Dependency working set retained for `problems/muse_attention_decode/dev/cluster_sliding/HANDOFF_LAST.md`, `problems/muse_attention_decode/dev/cluster_sliding/build.py`.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [HANDOFF.md](HANDOFF.md): Dependency retained by the live reference graph.
- [README.md](README.md): Current working set and resume instructions.
- [partial_port.py](partial_port.py): Fuse projection split reduction into final S11's single-launch attention port.
- [patch_partial_port.py](patch_partial_port.py): Apply the direct projection-partials attention port to frozen final S11.
- [vq8.cu](vq8.cu): Device source or payload template used by retained code.
- [out/partial_port.cu](out/partial_port.cu): Device source or payload template used by retained code.
- [out/s11_partial_port_decode1.py](out/s11_partial_port_decode1.py): Runtime or source-generation dependency.
- [out/vq8_c262e4c5.cu](out/vq8_c262e4c5.cu): Device source or payload template used by retained code.

Handoff: No independent integration handoff..

Pending H200 senders: none owned here; follow the consuming lane.

Next: Resume from the consuming lane; this directory has no separate pending experiment or promoted patch.
