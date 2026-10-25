# Full-attention handoff status

No promotion from this lane. H200cf79aa070ca349c6 closes the former N128
register-V patch: adjacent ST3 is neutral atB16 (.99639 vsS13), loses atB32
(1.06502), and loses to P5's existing N1289a23ca9a at both shapes. The old
S13 patch5ae1b320 is retained only as experimental evidence.

Active base is **S14 final f1c77ce6**. See softmax/README.md and
softmax/reset_admission.json. One eight-variant proj-host batch is held for
2026-10-26 09:00 KST, dispatch030864131ce8d08f; no judge job ID yet.

Experimental S14 patches: softmax/s14_mix25.patch **46069ac5** and
softmax/s14_ping25.patch **e5a93f0d**. Both apply cleanly onto exactf1c77ce6;
only VQ8_CUDA and its extension key change. They have no whole-layer H200 result
and are not promotion recommendations. Select a supported stage win first, then
run exact normal/forced layer screens and the paired whole-layer lab.
