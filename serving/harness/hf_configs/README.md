These are untrusted download receipts, never Python modules or remote model code.
Inspect them with an isolated interpreter (`python -I`) or as plain text.

`dflash2/config.json` is the released RadixArk DFlash2 configuration. The three
files under `muse_target/` contain Hugging Face access-denied responses, despite
their `.json` suffixes. They are **not usable configurations**. `make_tiny.py`
constructs its target configuration from the pinned SGLang Muse configuration
and model implementation, and the competition's public kernel specification.
It does not load or execute anything from this directory.
