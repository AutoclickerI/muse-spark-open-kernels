# timeline

S11 timeline data used by current boundary attribution; logical bytes and node residence do not measure DRAM duty.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [activity.cpp](activity.cpp): Device source or payload template used by retained code.

Handoff: S11 acda6c62; no kernel patch.

Pending H200 senders: none owned here; follow the consuming lane.

Next: Use c6bd22e2084c4809 as the existing fixture-specific timeline, and use the pending S14 boundary batch for new attribution.

Runtime data for `boundary_attribution/reconcile_s11.py` (receipt `c6bd22e2084c4809`; distinct fixtures from the lever model):

| Case | Stage | Span us | Completion increment us | Read MB |
|---|---|---:|---:|---:|
| d16S | b1 | 7.65 | 2.40 | 1.671 |
| d16S | b2 | 19.17 | 2.78 | 1.704 |
| d16F | b1 | 10.24 | 2.40 | 1.671 |
| d16F | b2 | 19.42 | 2.66 | 1.704 |
| d32S | b1 | 9.06 | 2.82 | 3.244 |
| d32S | b2 | 21.95 | 3.14 | 3.277 |
| d32F | b1 | 8.80 | 2.94 | 3.244 |
| d32F | b2 | 21.06 | 2.88 | 3.277 |
