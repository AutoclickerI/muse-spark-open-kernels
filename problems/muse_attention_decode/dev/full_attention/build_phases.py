"""Freeze S13's full-attention kernels and add phase-only instrumentation in this lane."""
import ast
import hashlib
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]
BASE = R / 'problems/muse_decode_layer/cand_r3_best7_s13_final.py'
src = BASE.read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('958f1258')
tree = ast.parse(src)
def literal(name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == name)
def segments(names):
    lines = src.splitlines(True)
    return '\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for name in names for n in tree.body if getattr(n, 'name', '') == name)
q1 = literal('ATTN_PORT_CUDA_SRC')
vq = literal('VQ8_CUDA')
assert hashlib.sha256(vq.encode()).hexdigest().startswith('c262e4c5')
(H / 'attn_base.cu').write_text(q1)
(H / 'vq8_base.cu').write_text(vq)
macros = r'''
#ifndef FULL_CLOCK
#define FULL_CLOCK 0
#endif
#ifndef GATHER_ONLY
#define GATHER_ONLY 0
#endif
#if FULL_CLOCK
#define FAMARK(t) const unsigned long long t=clock64()
#define FAADD(s,t) if((threadIdx.x&31)==0)fc[s]+=clock64()-(t)
__device__ __forceinline__ unsigned long long fa_time(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
struct FaTimer {
 unsigned long long *s,*d;
 __device__ FaTimer(unsigned long long*ss,unsigned long long*dd):s(ss),d(dd){
  if((threadIdx.x&31)==0){for(int i=0;i<16;++i)s[i]=0;s[0]=fa_time();s[1]=clock64();}
 }
 __device__ ~FaTimer(){if((threadIdx.x&31)==0){s[12]=clock64()-s[1];s[13]=fa_time();s[9]=s[14]?clock64()-s[14]:0;for(int i=0;i<16;++i)d[i]=s[i];}}
};
#else
#define FAMARK(t)
#define FAADD(s,t)
#endif
'''
q = q1.replace('namespace cg =', macros + '\nnamespace cg =', 1)
q = q.replace('bool need_mask, float scale_log2) {', 'bool need_mask, float scale_log2, unsigned long long* fc=nullptr) {\n  FAMARK(t_qk);', 1)
q = q.replace('  // Online softmax per row', '  FAADD(6,t_qk);FAMARK(t_sm);\n  // Online softmax per row', 1)
q = q.replace('  // O += P V.', '  FAADD(7,t_sm);FAMARK(t_pv);\n  // O += P V.', 1)
marker = '\n}\n\ntemplate <int L, int STAGES>\n__global__'
assert q.count(marker) == 1
q = q.replace(marker, '\n  FAADD(8,t_pv);' + marker, 1)
old = '  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;'
assert q.count(old) == 1
q = q.replace(old, old + '''
#if FULL_CLOCK
  __shared__ unsigned long long fas[4][16];
  unsigned long long* fc=fas[warp];
  const long long nr=(long long)p.batch*(p.hkv/2)*(p.ns/p.cs)*L*16;
  auto dst=reinterpret_cast<unsigned long long*>(p.part_ml+nr*2)+(((b*(p.hkv/2)+kvg)*p.ns+split)*4+warp)*16;
  FaTimer fa_guard(fc,dst);FAMARK(t_prep);
#else
  unsigned long long* fc=nullptr;
#endif
''', 1)
old = '  for (int i = 0; i < ntiles; ++i) {\n    cp_wait<STAGES - 2>();\n    __syncthreads();'
new = '''  FAADD(2,t_prep);FAMARK(t_loop);
  for (int i = 0; i < ntiles; ++i) {
    FAMARK(t_wait);
    cp_wait<STAGES - 2>();
    FAADD(3,t_wait);FAMARK(t_sync);
    __syncthreads();
    FAADD(4,t_sync);FAMARK(t_issue);'''
assert q.count(old) == 1
q = q.replace(old,new,1)
q = q.replace('    load_rows(nxt + 1, rows_next);', '    load_rows(nxt + 1, rows_next);\n    FAADD(5,t_issue);', 1)
q = q.replace('    compute_tile<L, NT>', '#if !GATHER_ONLY\n    compute_tile<L, NT>', 1)
q = q.replace('need_mask, scale_log2);', 'need_mask, scale_log2,fc);\n#endif', 1)
old = '  cp_wait<0>();\n  if (last &&'
assert q.count(old) == 1
q = q.replace(old, '''  FAADD(11,t_loop);
#if FULL_CLOCK
  if(lane==0){fc[10]=ntiles;fc[14]=clock64();}
#endif
  cp_wait<0>();
#if GATHER_ONLY
  return;
#endif
  if (last &&''', 1)
q = q.replace('true, scale_log2);', 'true, scale_log2,fc);', 1)
(H / 'attn_clock.cu').write_text(q)
# Existing vq8 phase clocks aggregate publish and issue. Split each without
# changing publication order or the ring protocol. Non-fused clock rows 25-28 are free.
v = vq.replace('      transpose_group(VR + slot * 8192, VT + slot * 8192, pw, lane);', '      CLK_MARK(tp);\n      transpose_group(VR + slot * 8192, VT + slot * 8192, pw, lane);\n      CLK_ADD(256,54,tp);CLK_MARK(tf);', 1)
v = v.replace('      CLK_ADD(256, 52, t0);', '      CLK_ADD(256,55,tf);\n      CLK_ADD(256, 52, t0);', 1)
v = v.replace('      page_fetch(p, S, wa, ring, ps, pw, lane);', '      CLK_ADD(256,56,t1);CLK_MARK(tpage);\n      page_fetch(p, S, wa, ring, ps, pw, lane);\n      CLK_ADD(256,57,tpage);', 1)
needle = 'row[24] = a[53];'
assert needle in v
v = v.replace(needle, needle + '\n    for(int k=0;k<4;++k)row[25+k]=a[54+k];', 1)
(H / 'vq8_clock.cu').write_text(v)
# The local driver uses the existing sm120 math shim, but the source is always
# the S13 frozen kernel, not the other lane's mutable vq8.cu.
drv = (R / 'problems/muse_attention_decode/dev/vq8/vq8.py').read_text()
drv = drv.replace("(Path(__file__).resolve().parent / 'vq8.cu').read_text()", repr(v))
drv = drv.replace('muse_vq8_', 'full_vq8_')
drv = drv.replace("extra_cuda_cflags=['-O3',", "extra_cuda_cflags=['-O3', '-std=c++20',")
(H / 'vq_runtime.py').write_text(drv)
runtime = '''"""Exact S13 q1 plan/ABI with clock-only scratch after part_ml."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
HQ,HKV,Q_SIZE=16,2,1024
_EXT={}
_COUNTERS={}
_SMS={}
'''
runtime += 'Q1_BASE='+repr(q1)+'\nQ1_CLOCK='+repr(q)+'\nCPP='+repr(literal('ATTN_PORT_CPP_SRC'))+'\n'
runtime += segments(('_attn_main_plan',))+'\n'+(H/'q1_runtime.py.inc').read_text()
(H / 'q1_runtime.py').write_text(runtime)
common=(R/'prof/lab/verify1_sliding_fa3_pair.py').read_text().split('\nimport triton\n',1)[0]
payload=common+'\nSOURCES='+repr({'q1':runtime,'vq':drv})+'\n'+(H/'phases_lab.py.inc').read_text()
(H/'phases_lab.py').write_text(payload)
print('Frozen full kernels and clock builds', H)
