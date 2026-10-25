"""N128 full attention, one KV head per 192-thread CTA; two resident CTAs/SM at stage4."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_p5_head_base',HERE.parent/'full128/build.py');base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)

def source():
    s=base.source()
    def rep(a,b):
        nonlocal s
        assert a in s,a
        s=s.replace(a,b)
    rep('__launch_bounds__(384, 1) vq8_main','__launch_bounds__(192, 2) vq8_main')
    rep('const int kvh = 2 * hp + hh;','const int kvh = hp;')
    rep('((2 * hp + hh) * 8 + (row & 7))','(hp * 8 + (row & 7))')
    rep('const int hh = __shfl_sync(0xffffffffu, threadIdx.x >> 7, 0);','const int hh = 0;')
    rep('(threadIdx.x >> 5) - 8','(threadIdx.x >> 5) - 4')
    rep('q = lane >> 3, c = lane & 7','q = lane >> 2, c = lane & 3')
    rep('head = c >> 2, cb = (c & 3) * 16','head = 0, cb = c * 16')
    rep('(int64_t)hp * 128 + c * 16','(int64_t)hp * 64 + c * 16')
    rep('16 * pw + 4*(i&3) + q + 64*(i>>2)','32 * pw + 8*(i&3) + q + 64*(i>>2)')
    # Two issuing warps each own32 keys per64-key slab, including every byte they transpose.
    a=s.index('DI void page_fetch(');b=s.index('template <int ST>\nDI void producer',a)
    s=s[:a]+'''DI void page_fetch(const VP& p,const Sched& S,Walk& w,int* ring,int slot,int pw,int lane) {
  int b=0,hp=0,seq=1,key0=0;walk_next(p,S,w,b,hp,seq,key0);
#pragma unroll
  for(int slab=0;slab<2;slab++) {
    const int kk=32*pw+lane+64*slab,key=min(key0+kk,max(seq-1,0));
    cp4(ring+slot*128+kk,p.pt+(int64_t)b*p.pages+(key>>p.lg));
  }
}

'''+s[b:]
    rep('for (int h = 0; h < 2; h++)','for (int h = 0; h < 1; h++)')
    rep('for(int slab=0;slab<2;slab++)transpose_slab(vr+4096*slab,vt+4096*slab,pw,lane);',
        'for(int slab=0;slab<2;slab++)for(int half=0;half<2;half++)transpose_slab(vr+4096*slab,vt+4096*slab,2*pw+half,lane);')
    rep('* 16384','* 8192')
    rep('Kn = Qs + 8192','Kn = Qs + 4096');rep('VTn = Qs + 16384','VTn = Qs + 8192');rep('Ks = Qs + 24576','Ks = Qs + 12288')
    rep('24576 + ST * 49152','12288 + ST * 24576')
    rep('mb_init(&full[tid], 4); mb_init(&empty[tid], 8)','mb_init(&full[tid], 2); mb_init(&empty[tid], 4)')
    rep('if (warp >= 8)','if (warp >= 4)');rep('if (tid == 256)','if (tid == 128)');rep('CLK_ADD(256,','CLK_ADD(128,')
    rep('dim3(384)','dim3(192)');rep('vq8_main<ST>, 384,','vq8_main<ST>, 192,')
    rep('(int)(HK / 2), (int)(B * (HK / 2))','(int)HK, (int)(B * HK)')
    rep('const int head = (2 * hp + hh) * 8 + (row & 7);','const int head = hp * 8 + (row & 7);')
    rep('dim3(2 * p.L * p.wpr, p.NS)','dim3(p.L * p.wpr, p.NS)')
    # Keep partial slots at128 rows: the unchanged allocator/clock ABI remains exact.
    assert 'ST * 16384' not in s and 'warp >= 8' not in s
    return s

if __name__=='__main__':
    s=source();p=HERE/'full_head.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
