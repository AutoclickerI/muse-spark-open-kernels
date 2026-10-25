# Experimental full verify N128 register-V

Status: locally screened; performance decision waits on H200 **cf79aa070ca349c6**.
This is not a promotion recommendation. Earlier N64 register-V variants lost.

`s13_ldsv128_experimental.patch` SHA-256 **5ae1b320efc3a35e2c8129d5dd683573dab6b78b2ecdb146ce22af773927ab38**
applies onto `problems/muse_decode_layer/cand_r3_best7_s13_final.py` **958f1258**.
Generated candidate **a67ff29d0a0ef25e01ef830e4da9f92338139a6c9f986ffd43efc536637c34ad**.
Regenerate with `patch_ldsv128.py`; `git apply --check` passes.

Only existing full verify16/32 vq8 rows change. Cached tiles have128 keys; each
CTA owns one KV head and contains128 compute plus32 gather threads. Adjacent
physical CTAs gather complementary64B halves of the same128B page row. Native
FP8 QK drains before PV's V-A bank becomes live. V remains raw in shared;
LDSM.T, shuffles and PRMT form its register-A fragments; FP8 P is stored in
shared with the matching key permutation. O is transposed once per portion.
Selected ST3 uses three physical CTAs per SM. Query/output precision is unchanged.

CUDA **84eddcfe74840075f8b7db3f545ac89f8cb2b3491eb5d778c785b3e1a364ccee**:
native main123–125 registers atS2/3/4, zero stack and spills, noC75 warnings.
Combine63 registers. ST3 uses70,656 dynamic plus4,096 static shared bytes,
allowing a modeled three CTAs/SM. This is not a throughput measurement.

Normal and forced whole-layer local_eval each pass **13 tests /12 captured
shapes /25 static checks**. In each mode, both changed rows execute eagerly5
times and in a captured call. `candidate_admission.json` pins source/patch
hashes, all static checks and path counts. The actual one-head N128 pipeline
runs locally atST3, including full80K B16/32, new-cache gates, graph replay and
FP32-partial ABI. The H200 lab's larger two-head N128 comparator explicitly uses
S13 as its local stand-in; H200 runs its frozen native source9a23ca9a.

FP32 projection partials sum and round toBF16 in the adapter; that cost must be
included in whole-layer pairing. The first local N128 build failed the unchanged
resource gate; scheduling QK before PV fixes the register liveness rather than
suppressing C7512/spills. See `ldsv128_initial_failure.json` and native proofs.

H200 lab compares S13, the measured P5 N128 winner9a23ca9a, independent/paired
one-head ST2/3/4 and the N64 paired form, three fresh seeds ×16 rotating cold
replays with A:A. After it lands, select only a supported win, then pair the
whole layer on identical inputs and many seeds. Decode quota is currently zero;
no whole-layer H200 result for this patch exists.
