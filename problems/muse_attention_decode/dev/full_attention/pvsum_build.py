"""S15 N128 denominator sums co-issued in the existing PV group."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];out=H/'pvsum';out.mkdir(exist_ok=True)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest().startswith('60254883')
tree=ast.parse(base);node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA')
original=ast.literal_eval(node.value)


def one(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)


HELPER=r'''
// Denominator is a PV accumulator: one FP8 P*ones M64N8 per K32 chunk,
// co-issued in the existing PV commit group. No additional commit or wait.
DI void pv_den(float (&d)[4],const uint32_t* a) {
#if USE_WGMMA
 extern __shared__ __align__(1024) uint8_t sm[];
 uint64_t b=desc(su(sm+CLKPAD+24576));pin(b);
 asm volatile("{.reg .pred q;setp.ne.b32 q,1,0;wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},%8,q,1,1;}"
   :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));
#else
 // Exact E4M3 operand sum; independent probe also runs the native RS form.
 float r[2]={};
#pragma unroll
 for(int i=0;i<4;i++){
  float2 x=__half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(a[i]&65535),__NV_E4M3));
  float2 y=__half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(a[i]>>16),__NV_E4M3));
  r[i&1]+=x.x+x.y+y.x+y.y;
 }
#pragma unroll
 for(int k=1;k<=2;k*=2)for(int i=0;i<2;i++)r[i]+=__shfl_xor_sync(0xffffffffu,r[i],k);
 d[0]+=r[0];d[1]+=r[0];d[2]+=r[1];d[3]+=r[1];
#endif
}
DI void pv_sum64(float (&o)[9][4],const uint32_t (&a)[8],const uint8_t* VT){
 pv(*reinterpret_cast<float(*)[8][4]>(o),a,VT);
#pragma unroll
 for(int j=0;j<2;j++)pv_den(o[8],a+4*j);
}
'''


def cuda(source=None):
    c=original if source is None else source
    c=one(c,'DI void pv128(float (&o)[8][4]',HELPER+'\nDI void pv128(float (&o)[9][4]')
    c=one(c,'wg_rs(o,a[4*j],a[4*j+1],a[4*j+2],a[4*j+3],va);',
          'wg_rs(*reinterpret_cast<float(*)[8][4]>(o),a[4*j],a[4*j+1],a[4*j+2],a[4*j+3],va);pv_den(o[8],a+4*j);')
    c=one(c,'pv(o,a0,VT);pv(o,a1,VT+4096);','pv_sum64(o,a0,VT);pv_sum64(o,a1,VT+4096);')
    c=c.replace('float (&O)[8][4]','float (&O)[9][4]').replace('float O[8][4], m[2]','float O[9][4], m[2]')
    c=one(c,'for (int e = 0; e < 4; e++) O[n][e] = 0.f;',
          'for (int e = 0; e < 4; e++) O[n][e] = 0.f;\n    for(int e=0;e<4;e++)O[8][e]=0.f;')
    c=one(c,'for (int n = 0; n < 8; n++) {\n      O[n][0] *= alpha[0];',
          'for (int n = 0; n < 9; n++) {\n      O[n][0] *= alpha[0];')
    c=one(c,'pv(O, pk, VN);','pv_sum64(O, pk, VN);')
    c=one(c,'float sum0 = 0.f, sum1 = 0.f;\n#pragma unroll','// Denominator accumulates with PV, after E4M3 packing.\n#pragma unroll')
    c=one(c,'      sum0 += p0;\n      sum1 += p1;','')
    c=one(c,'    l[rh] = l[rh] * alpha[rh] + (sum0 + sum1);','')
    c=one(c,'      float lt = l[rh];','      float lt = O[8][2*rh]*.25f;')
    c=one(c,'uint8_t* Ks = Qs + 24576;','uint8_t* Ks = Qs + 25600;')
    c=one(c,'  if (tid < ST) {',
          '  if(tid<32)*(uint4*)(sm+CLKPAD+24576+16*tid)=make_uint4(0x38383838,0x38383838,0x38383838,0x38383838);\n  proxy_fence();\n  if (tid < ST) {')
    c=one(c,'return CLKPAD + 24576 + ST * 49152;','return CLKPAD + 25600 + ST * 49152;')
    return c


def main():
    c=cuda();tag=hashlib.sha256(c.encode()).hexdigest()[:12]
    lines=base.splitlines(True);lines[node.lineno-1:node.end_lineno]=['VQ8_CUDA = '+repr(c)+'\n']
    s=''.join(lines).replace('muse_vq8_9a23ca9a98dd_','muse_vq8_'+tag+'_')
    compile(s,'s15_pvsum','exec');(out/'candidate.py').write_text(s);(out/'pvsum.cu').write_text(c)
    patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'))
    (out/'candidate.patch').write_text(patch)
    meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
        cuda_sha256=hashlib.sha256(c.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),promotion=False)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)


if __name__=='__main__':main()
