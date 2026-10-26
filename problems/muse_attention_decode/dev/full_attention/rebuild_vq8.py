"""Independent S15 operand-ready, register allocation and parity-softmax arms."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];out=H/'rebuild';out.mkdir(exist_ok=True)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('60254883')
def get(src,key):
 return ast.literal_eval(next(n.value for n in ast.parse(src).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key))
BASE=get(base,'VQ8_CUDA')
def edit(s,a,b):
 assert s.count(a)==1,(a[:120],s.count(a));return s.replace(a,b)

def ready(c,drop=False):
 c=edit(c,'uint8_t* VR, uint8_t* VT, uint64_t* full, uint64_t* empty,','uint8_t* VR, uint8_t* VT, uint64_t* kfull, uint64_t* full, uint64_t* empty,')
 needle=next(l for l in c.splitlines() if 'transpose_' in l and 'VR + slot' in l)
 c=edit(c,needle,'      proxy_fence();\n      if(lane==0)mb_arrive(&kfull[slot]);\n'+needle)
 # All function argument lists, after producer's distinct signature above.
 c=c.replace('uint8_t* VT, uint64_t* full, uint64_t* empty,','uint8_t* VT, uint64_t* kfull, uint64_t* full, uint64_t* empty,')
 c=edit(c,'uint64_t* full, uint64_t* empty) {','uint64_t* kfull, uint64_t* full, uint64_t* empty) {')
 c=edit(c,'  mb_wait(&full[(rr + 1) % ST], ((rr + 1) / ST) & 1);','  mb_wait(&kfull[(rr + 1) % ST], ((rr + 1) / ST) & 1);')
 pv='pv128' if 'DI void pv128' in c else 'pv'
 needle=f'  {pv}(O, pc, VT + (rr % ST) * '
 line=next(l for l in c.splitlines() if l.startswith(needle))
 c=edit(c,'  wg_commit();\n  wg_fence();\n'+line,'  wg_commit();\n  mb_wait(&full[rr % ST], (rr / ST) & 1);\n  wg_fence();\n'+line)
 for operand in ('pn','pc'):
  line=next(l for l in c.splitlines() if l.startswith(f'    {pv}(O, {operand}, VT + (rlast % ST)'))
  c=edit(c,'    wg_fence();\n'+line,'    mb_wait(&full[rlast % ST], (rlast / ST) & 1);\n    wg_fence();\n'+line)
 c=edit(c,'      mb_wait(&full[r % ST], (r / ST) & 1);','      mb_wait(&kfull[r % ST], (r / ST) & 1);')
 c=c.replace('VT, full, empty, hh,','VT, kfull, full, empty, hh,')
 c=edit(c,'__shared__ __align__(16) uint64_t full[ST], empty[ST];','__shared__ __align__(16) uint64_t kfull[ST], full[ST], empty[ST];')
 c=edit(c,'if (tid < ST) { mb_init(&full[tid], 4);','if (tid < ST) { mb_init(&kfull[tid],4); mb_init(&full[tid], 4);')
 c=edit(c,'producer<ST>(p, S, Ks, VR, VT, full, empty, pages);','producer<ST>(p, S, Ks, VR, VT, kfull, full, empty, pages);')
 c=edit(c,'consumer<ST>(p, S, Qs, Kn, VTn, Ks, VT, full, empty);','consumer<ST>(p, S, Qs, Kn, VTn, Ks, VT, kfull, full, empty);')
 if drop:
  # Each producer warp arrives V(rr) before it can publish K(rr+1).
  # All four K(rr+1) arrivals therefore imply all four V(rr) arrivals.
  # Final PV has no K(next) wait, so its explicit V(rlast) wait stays.
  c=edit(c,'  mb_wait(&full[rr % ST], (rr / ST) & 1);\n','')
 return c

def registers(c):
 mark='  if (warp >= 8) {'
 assert c.count(mark)==1
 return c.replace(mark,'''#if USE_WGMMA
  if(warp>=8)asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
#endif
'''+mark)

def parity(c):
 # Keep QK(j+1)/PV(j) overlap. Each probability tile updates its parity's
 # max/sum/O; rescale next O only after current PV has completed.
 a=c.index('template <int ST, bool MASK>\nDI void step');z=c.index('// The portion\'s last steps:',a)
 s=c[a:z];s=s.replace('float (&O)[8][4]','float (&O)[2][8][4]').replace('float (&m)[2], float (&l)[2]','float (&m)[2][2], float (&l)[2][2]')
 s=s.replace('template <int ST, bool MASK>','template <int ST, bool MASK, int CUR>')
 s=edit(s,'  float alpha[2];','  float alpha[2];\n  constexpr int cur=CUR,next=cur^1;')
 s=s.replace('protect(O);','protect(O[cur]);').replace('pv128(O,','pv128(O[cur],').replace('pv(O,','pv(O[cur],')
 s=s.replace('  protect(O[cur]);','  protect(O[0]);protect(O[1]);')
 s=s.replace('(sx, m, l, alpha,','(sx, m[next], l[next], alpha,').replace('rescale(O, alpha);','rescale(O[next], alpha);')
 s=s.replace('rescale(O[next], alpha);','rescale(O[next], alpha);protect(O[0]);protect(O[1]);')
 c=c[:a]+s+c[z:]
 a=c.index('template <int ST>\nDI void finish');z=c.index('template <int ST>\nDI void consumer',a)
 s=c[a:z].replace('float (&O)[8][4]','float (&O)[2][8][4]').replace('float (&m)[2]','float (&m)[2][2]').replace('float (&l)[2]','float (&l)[2][2]')
 s=s.replace('template <int ST>','template <int ST,int CUR>').replace('step<ST, true>','step<ST, true,CUR^1>')
 s=s.replace('protect(O);','protect(O[CUR]);').replace('pv128(O,','pv128(O[CUR],').replace('pv(O,','pv(O[CUR],')
 c=c[:a]+s+c[z:]
 a=c.index('template <int ST>\nDI void consumer');z=c.index('template <int ST>\n__global__',a);s=c[a:z]
 s=edit(s,'float O[8][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, alpha[2];',
   'float O[2][8][4], m[2][2]={{-INFINITY,-INFINITY},{-INFINITY,-INFINITY}}, l[2][2]={{0,0},{0,0}}, alpha[2];')
 s=edit(s,'for (int e = 0; e < 4; e++) O[n][e] = 0.f;','for (int e = 0; e < 4; e++) {O[0][n][e]=0.f;O[1][n][e]=0.f;}')
 # Only softmax calls in consumer; step/finish calls still receive both states.
 s=s.replace('(sx, m, l, alpha,','(sx, m[0], l[0], alpha,')
 calls=[l for l in s.splitlines() if 'step<ST, false>' in l]
 assert len(calls)==3
 s=s.replace(calls[0],calls[0].replace('step<ST, false>','step<ST, false,0>'))
 s=edit(s,calls[1],calls[1].replace('step<ST, false>','step<ST, false,1>'))
 calls=[l for l in s.splitlines() if 'finish<ST>' in l];assert len(calls)==2
 for line in calls:s=edit(s,line,'        if((nr-1)&1)'+line.strip().replace('finish<ST>','finish<ST,1>')+'\n        else '+line.strip().replace('finish<ST>','finish<ST,0>'))
 va=s.index('    if (virt) {\n      // The block\'s own');vz=s.index('    const int slot =',va)
 v=s[va:vz].replace('    if (virt) {','    if (virt) {\n     auto runnew=[&]<int CUR>(){')
 v=v.replace('m[0], l[0]','m[CUR], l[CUR]').replace('rescale(O, alpha);','rescale(O[CUR], alpha);').replace('protect(O);','protect(O[CUR]);').replace('pv(O, pk, VN);','pv(O[CUR], pk, VN);')
 assert v.endswith('    }\n');v=v[:-6]+'     };\n     if(nr&1)runnew.template operator()<1>();else runnew.template operator()<0>();\n    }\n';s=s[:va]+v+s[vz:]
 s=edit(s,'      float lt = l[rh];',
 '''      const float mx=fmaxf(m[0][rh],m[1][rh]);
      const float safe=mx==-INFINITY?0.f:mx;
      const float w0=ex2(m[0][rh]-safe),w1=ex2(m[1][rh]-safe);
      float lt = w0*l[0][rh]+w1*l[1][rh];''')
 s=s.replace('make_float2(O[n][2 * rh], O[n][2 * rh + 1])','make_float2(w0*O[0][n][2*rh]+w1*O[1][n][2*rh],w0*O[0][n][2*rh+1]+w1*O[1][n][2*rh+1])')
 s=s.replace('make_float2(m[rh], lt)','make_float2(mx, lt)')
 c=c[:a]+s+c[z:];return c


def candidate(c):
 n=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA');lines=base.splitlines(True);lines[n.lineno-1:n.end_lineno]=['VQ8_CUDA = '+repr(c)+'\n'];s=''.join(lines)
 return s.replace('muse_vq8_9a23ca9a98dd_','muse_vq8_'+hashlib.sha256(c.encode()).hexdigest()[:12]+'_')


def main():
 meta={}
 for name,fn in [('ready',ready),('ready_elide',lambda c:ready(c,True)),('reg',registers),('parity',lambda c:registers(parity(c)))]:
  c=fn(BASE);s=candidate(c);(out/(name+'.cu')).write_text(c);(out/(name+'.py')).write_text(s)
  # Local launch stand-in retains this exact N128 arithmetic/state logic.
  from local_n128 import source as local_source
  lc=local_source(c)
  patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'));(out/(name+'.patch')).write_text(patch)
  meta[name]=dict(source_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(c.encode()).hexdigest(),local_standin_sha256=hashlib.sha256(lc.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
 (out/'vq8_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
if __name__=='__main__':main()
