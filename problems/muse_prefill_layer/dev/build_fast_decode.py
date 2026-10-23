"""Adapt the MoE owner's lossless bit-plane encoding to prefill WGMMA fragments."""
from pathlib import Path
import re
p=Path(__file__).resolve().parents[1]
s=(p/'dev/moe_w4a16.cu').read_text()
i=s.index('__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2');j=s.index('\n__device__ __forceinline__ uint32_t bmul',i)
s=s[:i]+'''// Encoded scale byte u: |sf| = 2^(u/8 - 21) * (1 + (u%8)/8).
// Reconstruct |sf| * 2^118 exactly in BF16. Zero has its own code.
__device__ __forceinline__ __nv_bfloat162 e4m3x2_to_bf16x2(uint32_t two_bytes) {
  uint32_t lo = two_bytes & 255, hi = (two_bytes >> 8) & 255;
  uint32_t bits = (lo ? 0x7000u + (lo << 4) : 0) | ((hi ? 0x7000u + (hi << 4) : 0) << 16);
  return *reinterpret_cast<__nv_bfloat162*>(&bits);
}
'''+s[j:]
i=s.index('__device__ __forceinline__ void make_frag(');j=s.index('\n__device__ __forceinline__ uint64_t make_desc_b',i)
s=s[:i]+'''// Sixteen re-encoded E2M1 codes in A,B -> eight BF16x2 words.
// The bit planes represent FP4 * 2^-126; multiplication by the encoded scale
// yields the exact NVFP4 weight / 256. The epilogue restores the power of two.
__device__ __forceinline__ void make_frag(uint32_t A, uint32_t B, int half,
      __nv_bfloat162 sf, uint32_t lo_a, uint32_t hi_a, uint32_t (&a)[4]) {
  constexpr uint32_t MASK = 0x81C081C0u, LA = 0x60306030u;
  uint32_t r[4];
  if (half == 0) {
    r[0] = A & MASK;
    r[1] = __funnelshift_r(A,A,4) & MASK;
    r[2] = __funnelshift_r(A,A,10) & MASK;
    r[3] = __funnelshift_r(B,B,2) & MASK;
  } else {
    r[0] = __funnelshift_r(B,B,6) & MASK;
    r[1] = __funnelshift_r(B,B,12) & MASK;
    uint32_t W = (A & LA) | (B & ~LA);
    r[2] = __funnelshift_r(W,W,24) & MASK;
    r[3] = __funnelshift_r(W,W,30) & MASK;
  }
  __nv_bfloat162 slo = __low2bfloat162(sf), shi = __high2bfloat162(sf);
  a[0] = bmul(r[0], slo); a[1] = bmul(r[1], shi);
  a[2] = bmul(r[2], slo); a[3] = bmul(r[3], shi);
}
'''+s[j:]
for old,new in [('raw[j].x, e4m3','raw[j].x, raw[j].y, 0, e4m3'),('raw[j].y, e4m3','raw[j].x, raw[j].y, 1, e4m3'),('raw[j].z, e4m3','raw[j].z, raw[j].w, 0, e4m3'),('raw[j].w, e4m3','raw[j].z, raw[j].w, 1, e4m3')]:
 # Match only the original argument: earlier replacements must not match again.
 s=s.replace('make_frag('+old, 'make_frag('+new)
s=s.replace('const float gsc = p.gs[e];', 'const float gsc = p.gs[e] * 256.f;')
(p/'dev/moe_w4a16_fast.cu').write_text(s)
# Generate the bit permutation inverse to the four rotate/mask extracts.
slots=(('A',0),('A',4),('A',10),('B',2),('B',6),('B',12),('W',24),('W',30))
la=0x60306030;table=[];used={'A':set(),'B':set()}
for k,(word,rot) in enumerate(slots):
 for h in range(2):
  for bit,db in {3:15,2:8,1:7,0:6}.items():
   pos=(db+16*h+rot)%32;w=word if word!='W' else ('A' if (la>>pos)&1 else 'B')
   assert pos not in used[w];used[w].add(pos);table.append((2*k+h,bit,w,pos))
assert all(len(v)==32 for v in used.values())
helper='''
@triton.jit
def _encode_blob_fast(B, S, O):
    pid = tl.program_id(0).to(tl.int64)
    t = tl.arange(0,128)
    slot = (t//32)*8 + (t%32)//4
    for pair in tl.static_range(2):
        x = tl.load(B + pid*512 + t*4 + pair*2).to(tl.uint32)
        y = tl.load(B + pid*512 + t*4 + pair*2+1).to(tl.uint32)
        s0 = tl.load(S + pid*256 + slot*8 + pair*4).to(tl.int32)
        s1 = tl.load(S + pid*256 + slot*8 + pair*4+1).to(tl.int32)
        s2 = tl.load(S + pid*256 + slot*8 + pair*4+2).to(tl.int32)
        s3 = tl.load(S + pid*256 + slot*8 + pair*4+3).to(tl.int32)
'''
for q in range(16):
 sr=f's{2*(q//8)+(q//2)%2}';raw='x' if q<8 else 'y'
 helper+=f'        c{q} = (({raw} >> {4*(q%8)}) & 15) ^ tl.where(({sr} & 128) != 0, 8, 0).to(tl.uint32)\n'
 helper+=f'        c{q} = tl.where(({sr} & 127) != 0, c{q}, 0)\n'
for word in ('A','B'):
 terms=[f'(((c{q} >> {b}) & 1) << {pos})' for q,b,w,pos in table if w==word]
 helper+='        o'+word+' = '+' | '.join(terms)+'\n'
helper+='''        tl.store(O + pid*512 + t*4 + pair*2, oA)
        tl.store(O + pid*512 + t*4 + pair*2+1, oB)

@triton.jit
def _encode_sf_fast(S, O, N:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    a = tl.load(S+i,i<N,0).to(tl.int32)&127
    u = a+112
    u = tl.where(a==1,96,u); u=tl.where(a==2,104,u); u=tl.where(a==3,108,u)
    u = tl.where(a==4,112,u); u=tl.where(a==5,114,u); u=tl.where(a==6,116,u); u=tl.where(a==7,118,u)
    tl.store(O+i,tl.where(a==0,0,u).to(tl.uint8),i<N)

def repack_fast(w,sf,gated):
    b,s = repack(w,sf,gated)
    o=torch.empty_like(b); so=torch.empty_like(s)
    _encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),s,o.view(torch.int32),num_warps=4)
    _encode_sf_fast[(triton.cdiv(s.numel(),1024),)](s,so,s.numel(),BLOCK=1024,num_warps=4)
    return o,so
'''
(p/'dev/fast_decode_repack.py').write_text(helper)
v=(p/'cand_v3.py').read_text()
v=re.sub(r'^CUDA_SRC = .*$',lambda m:'CUDA_SRC = '+repr(s),v,flags=re.M)
v=v.replace('name="muse_prefill_grouped_w4a16"','name="muse_prefill_grouped_w4a16_fast"')
i=v.index('    try:\n        st["bank"] = _marlin_bank(weights)');j=v.index('    torch.cuda.synchronize()',i)
v=v[:i]+'    st["bank"] = None\n'+v[j:]
v=v.replace("= repack(weights[", "= repack_fast(weights[")
v=v.replace("    st['sms'] =", "    for key in ('w13', 'w13_sf', 'w2', 'w2_sf'):\n        st.pop(key, None)\n    st['sms'] =")
v+=helper
v=v.replace('Muse Spark Open prefill layer: v1.', 'Muse Spark Open prefill layer: v17, lossless bit-plane NVFP4 decoding.')
out=p/'cand_v17.py'
if out.exists():raise SystemExit('Refusing to overwrite '+str(out))
out.write_text(v)
print(out)
