"""Pin each review premise to the exact S15 code and current measured preconditions."""
import hashlib,json
from pathlib import Path
from build import original
D=Path(__file__).resolve().parent;R=D.parents[4];source,cell,cuda=original()
assert 'const int target = NT1 * (SLOT ? 1 : rt->nblk[e]);' in cuda
assert 'while (ld_acquire(cnt + C_DONE + e) < target)' in cuda
assert cuda.count('atomicAdd(cnt + C_DONE + cur_e, seg_tiles)')==2
assert 'v *= sc;' in cuda and 'const float sc = gs2[e] * 256.f;' in cuda
assert 'rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f)' in cuda
assert 'const float z = v * fw[0];' in cuda
assert 'o[0] = A & DQ_MASK;' in cuda and 'hmul2(os[0], Sr)' in cuda
cases={
 'a_BLOCK':dict(confirmed=True,code='SMALLM_VAR_CUDA::unit_body; expert-wide target NT1*nblk[e], two fence/barrier/atomic publication sites',implementation='Index all publications/acquires by e*4+nb, target NT1;128 done counters plus3 existing control slots;512B done region. Same store/fence/publish/acquire order. Dispatch only M16/M32.'),
 'b_CANCEL':dict(confirmed=True,code='SMALLM_VAR_CUDA::W2 epilogue v*=256*gs2[e], then z=v*(1+pnw), ss+=z*z, final RMSNorm epsilon1e-8',identity='a*u/sqrt(a*a*mean(u*u)+eps) = sign(a)*u/sqrt(mean(u*u)+eps/(a*a)); a=0 produceszero. W13 is untouched because SwiGLU intervenes.',implementation='Remove W2 element multiply; prepare96 FP32 norm-metadata values, retaining original32 FP32 global scales. Finalizer uses eps/a^2 and sign; double evaluation for epsilon overflow/underflow cases; zero scale/zero norm guarded.',extreme_tests='extreme_validation.json:160 helper cases, including signed/tiny/huge scales;component_validation.json:zero/negative and1e-10/1e10 full native MoE cases'),
 'c_SHARDS':dict(confirmed=True,code='H200-gated PAIR flush_vec_norm uses a single atomic ss+off+n; LOCAL exports one CTA partial per pair and adds finalizer gather',implementation='Only replace PAIR norm slot by2/4/8 c-indexed shards. Each W13 tile0 zeroes all shards before publication; finalizer sums shards. No CTA-local export/gather.'),
 'd_TWO_BASE':dict(confirmed=True,code='S15 uses recoded dq8 sign/magnitude bit extraction plus128 HMUL2 per math tile; the finite scalar identity S/1.5S is exact',precondition='audit_decode.py:8,128 finite signed-format packed-half checks. Same16 output BF16x2 registers per lane in old/new probes.',decision='Dropped before H200: sm90a integer instructions37->406 and registers30->31 for this implementation.',leaves_open='A more efficient packed exponent decoder; this loss does not close all two-base methods.'),
 'e_PACK5':dict(confirmed=True,code='S15 tile has32,768 packed-code bytes plus4,096 exact recoded scale bytes; prefix tables retain FP4 and scale precision',measurement='Official complete layer weight tensors from generate_weights(seed5353):100,663,296 scale bytes;every256-byte packet fits a32-symbol window. Final32B tile header and8B offset accounting:4.05816% byte cut.',implementation='256-scale packets:one base byte,5-bit deltas or raw256 bytes. Per-tile raw mask,aligned payload and64-bit tile offsets;timed TMA/decode. Wide raw transfer and full-bank code/scale equality verified.',leaves_open='Actual H200 benefit after decoding and metadata cost; packet widths and layouts can differ.')}
report=dict(S15_source=str(Path('problems/muse_decode_layer/cand_r3_best7_s15.py')),S15_sha256=hashlib.sha256(source.encode()).hexdigest(),SMALLM_VAR_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),claims=cases)
(D/'claims.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
