# Tile-contiguous weight layout for the small-M GEMVs

Question: is the GEMV's gap to a sequential read (`../gemv_bal/README.md`) DRAM locality? Each CTA of S8's
GEMV reads BN rows x 256 B per stage, 16 KB apart; the read kernel streams consecutive addresses.
`lab_body.py` + `build.py` -> `prof/lab/gemv_tile.py`: the same balanced split-K kernel on the row-major
weight (`r`) and on a repacked copy [N/BN, K/BK, BN, BK] (`t`: each stage one contiguous 8-16 KB block).
H200: send 75433cd18d5c1ef8. A win would need every M of that role on custom kernels: a BF16 repack must
replace the weight, not sit beside it.
- H200 `a192785286ac47fa`: the tile-contiguous copy gains only 0.1-0.35 us over the same balanced kernel on the row-major weight (e.g. pool_router t=16 20.98 vs 21.33; unpooler t=16 17.94 vs 18.27). Not pursued.
