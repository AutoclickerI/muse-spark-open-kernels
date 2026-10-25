"""Native N128 cached tiles, two KV heads per stream-K CTA, N64 virtual tile and unchanged combine."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_p5_cluster_source',HERE.parent/'cluster_sliding/build.py')
cluster_build=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(cluster_build)

def source():
    old=(HERE.parent/'vq8/out/vq8_c262e4c5.cu').read_text();s=old.replace('constexpr int NSMAX = 128','constexpr int NSMAX = 64')
    base=cluster_build.source();a=base.index('DI void ss128(');b=base.index('DI void rescale(',a)
    ops=base[a:b]
    helpers='''template<int N> DI void protect(float (&x)[N][4]) {
#pragma unroll
 for(int i=0;i<N;i++)for(int e=0;e<4;e++)asm volatile("":"+f"(x[i][e])::"memory");
}
template<int N> DI void protect(uint32_t (&x)[N]) {
#pragma unroll
 for(int i=0;i<N;i++)asm volatile("":"+r"(x[i])::"memory");
}
DI void pv128(float (&o)[8][4],const uint32_t (&a)[16],const uint8_t* VT) {
  uint32_t a0[8],a1[8];
#pragma unroll
  for(int i=0;i<8;i++){a0[i]=a[i];a1[i]=a[i+8];}
  pv(o,a0,VT);pv(o,a1,VT+4096);
}
'''
    # Avoid live RS operand copies in native WGMMA: use the registers directly.
    helpers=helpers[:helpers.index('DI void pv128')]+'''DI void pv128(float (&o)[8][4],const uint32_t (&a)[16],const uint8_t* VT) {
#if USE_WGMMA
#pragma unroll
  for(int j=0;j<4;j++) {uint64_t va=desc(su(VT+(j/2)*4096))+(j%2)*2;pin(va);wg_rs(o,a[4*j],a[4*j+1],a[4*j+2],a[4*j+3],va);}
#else
  uint32_t a0[8],a1[8];
#pragma unroll
  for(int i=0;i<8;i++){a0[i]=a[i];a1[i]=a[i+8];}
  pv(o,a0,VT);pv(o,a1,VT+4096);
#endif
}
'''
    p=s.index('// N (token, col) pairs');s=s[:p]+ops+helpers+s[p:]
    # Reuse the existing softmax with a compile-time N-block count (8 virtual, 16 cached).
    a=s.index('template <bool MASK>\nDI void softmax');b=s.index('DI void rescale(',a)
    sm=s[a:b].replace('template <bool MASK>', 'template <int NB,bool MASK>').replace('float (&s)[8][4]','float (&s)[NB][4]').replace('uint32_t (&pk)[8]','uint32_t (&pk)[NB]')
    sm=sm.replace('n < 8','n < NB').replace('float v[8]','float v[NB]')
    sm=sm.replace('for (int n = 0; n < 4; n++) v[n] = fmaxf(v[n], v[n + 4]);', 'for (int stride=NB/2;stride;stride/=2)\n#pragma unroll\n      for(int n=0;n<stride;n++)v[n]=fmaxf(v[n],v[n+stride]);')
    sm=sm.replace('mx[rh] = fmaxf(fmaxf(v[0], v[2]), fmaxf(v[1], v[3]));','mx[rh] = v[0];').replace('kk < 2','kk < NB/4')
    s=s[:a]+sm+s[b:]
    s=s.replace('(seq - lo + 63) >> 6','(seq - lo + 127) >> 7').replace('(sq - lo + 63) >> 6','(sq - lo + 127) >> 7').replace('w.lo + 64 * w.k','w.lo + 128 * w.k')
    # Two 64-key transpose slabs for each KV head.
    a=s.index('DI void transpose_group(');b=s.index('struct Walk',a)
    tr=s[a:b].replace('transpose_group','transpose_slab').replace('h * 4096','h * 8192')
    tr+='DI void transpose_group(const uint8_t* vr,uint8_t* vt,int pw,int lane) {\n#pragma unroll\n  for(int slab=0;slab<2;slab++)transpose_slab(vr+4096*slab,vt+4096*slab,pw,lane);\n}\n'
    s=s[:a]+tr+s[b:]
    a=s.index('DI void page_fetch(');b=s.index('// Online softmax',a)
    pp=s[a:b].replace('16 * pw + (lane & 15)','32 * pw + lane').replace('if (lane < 16) cp4(ring + slot * 64 + 16 * pw + lane','cp4(ring + slot * 128 + 32 * pw + lane')
    pp=pp.replace('slot * 8192','slot * 16384').replace('head * 4096','head * 8192').replace('i < 4','i < 8').replace('16 * pw + 4 * i + q','32 * pw + 4 * i + q').replace('ps * 64 + kk','ps * 128 + kk')
    s=s[:a]+pp+s[b:]
    a=s.index('template <int ST, bool MASK>\nDI void step');b=s.index('template <int ST>\n__global__',a)
    c=s[a:b].replace('float (&sx)[8][4]','float (&sx)[16][4]').replace('uint32_t (&pc)[8]','uint32_t (&pc)[16]').replace('uint32_t (&pn)[8]','uint32_t (&pn)[16]')
    # Only cached calls change, while the virtual tile remains N64.
    c=c.replace('qk(sx, Q, Ks','qk128(sx, Q, Ks').replace('pv(O, pc, VT','pv128(O, pc, VT').replace('pv(O, pn, VT','pv128(O, pn, VT')
    c=c.replace('* 8192 + hh * 4096','* 16384 + hh * 8192').replace('uint32_t pa[8], pb[8]','uint32_t pa[16], pb[16]')
    c=c.replace('      float sx[8][4];\n      mb_wait','      float sx[16][4];\n      mb_wait')
    c=c.replace('softmax<MASK>','softmax<16,MASK>').replace('softmax<true>','softmax<16,true>').replace('softmax<false>','softmax<16,false>').replace('lo + 64 *','lo + 128 *')
    c=c.replace('softmax<16,true>(sx, m, l, alpha, pk','softmax<8,true>(sx, m, l, alpha, pk')
    s=s[:a]+c+s[b:]
    s=s.replace('Ks + ST * 8192','Ks + ST * 16384').replace('VR + ST * 8192','VR + ST * 16384').replace('int pages[pd_of<ST>() * 64]','int pages[pd_of<ST>() * 128]')
    s=s.replace('24576 + ST * 24576','24576 + ST * 49152')
    # Higher stage counts cannot fit. Keep resource queries honest and compile only ST2/3/4.
    s=s.replace('case 6: launch<6>','case 3: launch<3>').replace('    case 8: launch<8>(p, st, pdl); break;\n','')
    s=s.replace('if (stages == 6) return resources<6>();\n  return resources<8>();','if (stages == 3) return resources<3>();\n  return resources<4>();')
    assert 'softmax<true>' not in s and 'softmax<false>' not in s
    return s

if __name__=='__main__':
    s=source();p=HERE/'full128.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
