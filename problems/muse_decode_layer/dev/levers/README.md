# levers

Source-verified S11 lever analysis and current ranking.

Files (one entry per retained direct file):

- [README.md](README.md): Current working set and resume instructions.
- [build.py](build.py): Offline, source-pinned 12-shape lever model; refresh from source-verified incumbent profiles.
- [current.json](current.json): Current identity, measurement or validation data consumed by retained code.
- [scenarios.py](scenarios.py): Decode-layer score against stage floors (docs/reference/bound.md "Score vs floor").

Handoff: [build.py](build.py), SHA256 `ecc2a3adc971aa660bdc40ea3638b619ba5f630657a11abe23972e769e1f3ec0`, applies onto S11 acda6c62 analysis; no kernel patch.

Pending H200 senders: none owned here; follow the consuming lane.

Next: Rebuild current.json against the next promoted incumbent and source-verified profile; distinguish PDL residence from critical-path work. Evidence: 863f502c62e54d7b.
