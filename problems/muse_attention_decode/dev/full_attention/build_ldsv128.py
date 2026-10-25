"""N128/one-head occupancy with raw V and matrix-load register-A PV."""
import ast
import hashlib
import re
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H.parent/'full128/full128.cu').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='9a23ca9a98ddeaff252bad309d0f1c774b0aa0f72144c6fe4f74a39dddf45507'
frozen128=s
base=(H/'ldsvone.cu').read_text()
helpers=base[base.index('DI uint8_t* prob_shared()'):base.index('// N (token, col)')]
helpers=helpers.replace('sm+CLKPAD+16384','sm+CLKPAD+20480')
helpers=helpers.replace('DI void pv(float (&o)[8][4],uint32_t (&a)[8]', 'template<int NB> DI void pv_any(float (&o)[8][4],uint32_t (&a)[NB]')
helpers=helpers.replace('kk<2','kk<NB/4')
helpers=helpers.replace('col=32*kk+16*hf+4*t','col=32*(kk%2)+16*hf+4*t')
helpers=helpers.replace('ps+off(row,col)', 'ps+4096*(kk/2)+off(row,col)').replace('ps+off(row,col+2)','ps+4096*(kk/2)+off(row,col+2)')
a=helpers.index(' uint64_t pa=desc');b=helpers.index('#else',a)
helpers=helpers[:a]+'''#pragma unroll
 for(int kk=0;kk<NB/4;++kk){
  uint64_t pa=desc(su(ps+4096*(kk/2)))+2*(kk%2);pin(pa);
  wg_rs(o,a[4*kk],a[4*kk+1],a[4*kk+2],a[4*kk+3],pa);
 }
'''+helpers[b:]
helpers=helpers.replace('word(ps,8*n+g,32*kk+4*t)','word(ps+4096*(kk/2),8*n+g,32*(kk%2)+4*t)').replace('word(ps,8*n+g,32*kk+16+4*t)','word(ps+4096*(kk/2),8*n+g,32*(kk%2)+16+4*t)')
helpers+='''DI void pv(float (&o)[8][4],uint32_t (&a)[8],const uint8_t* v){pv_any<8>(o,a,v);}
DI void pv128(float (&o)[8][4],uint32_t (&a)[16],const uint8_t* v){pv_any<16>(o,a,v);}
'''
a=s.index('// O[64 x 64 dims]');b=s.index('DI void ss128(',a)
s=s[:a]+helpers+'\n'+s[b:]
# The frozen N128 source's old P-register PV helper is replaced by the above.
a=s.index('DI void pv128(',s.index('DI void ss128('));b=s.index('// N (token, col)',a)
s=s[:a]+s[b:]
old='for (int i = 0; i < 8; i++) pk[i] = 0u;\n    return;'
assert s.count(old)==1;s=s.replace(old,old.replace('i < 8','i < NB').replace('    return;','    save_alpha(alpha);\n    return;'))
mark='  // A fragment for PV chunk kk:';assert s.count(mark)==1;s=s.replace(mark,'  save_alpha(alpha);\n'+mark)
a=s.index('DI void rescale(');b=s.index('// A portion',a)
s=s[:a]+'DI void rescale(float (&O)[8][4],const float (&alpha)[2]){}\n\n'+s[b:]
s=s.replace('      transpose_group(VR + slot * 16384, VT + slot * 16384, pw, lane);','      // No shared V transpose; consumers form A with LDSM.T.')
old='''    const int pos = 4 * (tok >> 1) + (tok & 1);
    VN[off(2 * lane, pos)] = (uint8_t)(vq & 0xff);
    VN[off(2 * lane + 1, pos)] = (uint8_t)(vq >> 8);'''
assert s.count(old)==1;s=s.replace(old,'    *(uint16_t*)(VN+off(tok,2*lane))=vq;')
old='    const int slot = S.partB[s] + (cta - cta_of(S.segP[s], G, T));'
assert s.count(old)==1;s=s.replace(old,'    restore_output(O,Qs,hh);\n'+old)
# One head/CTA:64 live query rows and one producer warp over128 keys.
s=s.replace('(int)(HK / 2), (int)(B * (HK / 2))','(int)HK, (int)(B * HK)')
s=s.replace('2 * hp + hh','hp')
s=re.sub(r'(?:const )?int hh\s*=\s*__shfl_sync\(0xffffffffu,\s*threadIdx.x\s*>>\s*7,\s*0\);','const int hh=0;',s)
s=s.replace('* 128 * PR','* 64 * PR').replace('slot * 128 +','slot * 64 +').replace('pb * 128 +','pb * 64 +')
s=s.replace('slot * 16384','slot * 8192').replace('% ST) * 16384','% ST) * 8192').replace('hh * 8192','hh * 4096')
s=s.replace('(int64_t)hp * 128 + c * 16','(int64_t)hp * 64 + c * 16')
s=s.replace('(threadIdx.x >> 5) - 8','(threadIdx.x >> 5) - 4').replace('lane >> 3, c = lane & 7','lane >> 2, c = lane & 3')
s=s.replace('const int head = c >> 2, cb = (c & 3) * 16;','const int head = 0, cb = c * 16;')
s=s.replace('for (int i = 0; i < 8; i++) {\n        const int kk = 16 * pw + 4*(i&3) + q + 64*(i>>2);','for (int i = 0; i < 16; i++) {\n        const int kk = 8*i+q;')
assert 'const int kk = 8*i+q;' in s
a=s.index('  const int key = min(key0 + 16 * pw');b=s.index('\n}',a)
s=s[:a]+'''#pragma unroll
 for(int j=0;j<4;++j){int key=min(key0+32*j+lane,max(seq-1,0));
  cp4(ring+slot*128+32*j+lane,p.pt+(int64_t)b*p.pages+(key>>p.lg));}
'''+s[b:]
s=s.replace('__launch_bounds__(384, 1)','__launch_bounds__(160, 3)')
s=s.replace('uint8_t* Kn = Qs + 8192;','uint8_t* Kn = Qs + 4096;').replace('uint8_t* VTn = Qs + 16384;','uint8_t* VTn = Qs + 8192;')
s=s.replace('uint8_t* Ks = Qs + 24576;','uint8_t* Ks = Qs + 21504;').replace('uint8_t* VR = Ks + ST * 16384;','uint8_t* VR = Ks + ST * 8192;').replace('uint8_t* VT = VR + ST * 16384;','uint8_t* VT = VR;')
s=s.replace('mb_init(&full[tid], 4); mb_init(&empty[tid], 8);','mb_init(&full[tid], 1); mb_init(&empty[tid], 4);').replace('if (warp >= 8)','if (warp >= 4)')
s=s.replace('24576 + ST * 49152','21504 + ST * 16384')
s=s.replace('cfg.blockDim = dim3(384);','cfg.blockDim = dim3(160);').replace('vq8_main<ST>, 384, smem','vq8_main<ST>, 160, smem').replace('dim3(2 * p.L * p.wpr, p.NS)','dim3(p.L * p.wpr, p.NS)')
# A 64-register QK accumulator cannot overlap PV's32 O +16 A registers
# at a128-register limit. Drain QK and pack P before forming V's A bank.
# Three resident head CTAs provide the independent work instead.
a=s.index('    if (nr > 0) {',s.index('DI void consumer'))
b=s.index('    r += nr;',a)
s=s[:a]+'''#pragma unroll 1
    for(int j=0;j<nr;++j){
      float sx[16][4];uint32_t pc[16];
      mb_wait(&full[(r+j)%ST],((r+j)/ST)&1);
      protect(sx);wg_fence();qk128(sx,Q,Ks+((r+j)%ST)*8192);
      wg_commit();wg_wait<0>();protect(sx);
      if(ka+j==nc-1)softmax<16,true>(sx,m,l,alpha,pc,sc,lo+128*(ka+j),lo_r,hi_r,live);
      else softmax<16,false>(sx,m,l,alpha,pc,sc,lo+128*(ka+j),lo_r,hi_r,live);
      pv128(O,pc,VT+((r+j)%ST)*8192);
      wg_commit();wg_wait<0>();protect(O);protect(pc);
      if(lane==0)mb_arrive(&empty[(r+j)%ST]);
    }
'''+s[b:]
ca=s.index('  float2 gate = make_float2',s.index('vq8_combine(VP p)'));cb=s.index("  // This warp's slice",ca)
gate=s[ca:cb];s=s[:ca]+s[cb:]
mark='  *(__nv_bfloat162*)(p.out + ((int64_t)(b * p.L + tok) * p.HQ + head) * 64 + 2 * lane) ='
assert s.count(mark)==1;s=s.replace(mark,gate+mark)
a=s.index('template <int ST> constexpr int smem_bytes()');s=s[:a].replace('p.L','8').replace('p.window','0').replace('p.proj_splits','0')+s[a:]
s=s.replace('if (HK % 2 || HQ != 8 * HK || p.NS > NSMAX || (L != 1 && L != 8))','if (HQ != 8 * HK || p.NS > NSMAX || L != 8 || window || proj_splits)')
probe=base[base.index('__global__ void regv_probe'):].replace('sm+17408','sm+21504').replace('<<<1,128,21504,','<<<1,128,25600,')
s+='\n'+probe
(H/'ldsv128.cu').write_text(s)
tree=ast.parse((H/'onehead.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
drv=ast.unparse(tree)+'\n';(H/'ldsv128.py').write_text(drv)
test=(H/'check_onehead.py').read_text().replace("load('onehead')","load('ldsv128')").replace("'onehead.py'","'ldsv128.py'").replace("'onehead_validation.json'","'ldsv128_validation.json'").replace('stages=6','stages=3').replace('setting=(6,1)','setting=(3,1)')
(H/'check_ldsv128.py').write_text(test)
print('N128 matrix-load one-head:',hashlib.sha256(s.encode()).hexdigest())
tree=ast.parse((H/'vq_runtime.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(frozen128)
bestdrv=ast.unparse(tree)+'\n'
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
lab=(H/'hybrid_lab.py.inc').read_text().replace('range(5)','range(7)').replace('range(3)','range(5)')
lab=lab.replace("    for n,s in SOURCES.items():",'''    for n,s in SOURCES.items():
        # The comparator's two-head N128 ring exceeds sm120's per-CTA shared limit.
        # Local boot uses the exact S13 ABI stand-in; H200 compiles/runs frozen9a23ca9a.
        if n=='best128' and torch.cuda.get_device_capability()[0]!=9:s=SOURCES['old']''')
old="arms=[old,old]+[lambda st=st:mods['new']['vq8_attention'](d,stages=st,min_tiles=1) for st in (4,6,8)]"
assert lab.count(old)==1
lab=lab.replace(old,"arms=[old,old,lambda:mods['best128']['vq8_attention'](d,stages=4,min_tiles=1)]+[lambda st=st:mods['new']['vq8_attention'](d,stages=st,min_tiles=1) for st in (2,3,4)]+[lambda:mods['pair64']['vq8_attention'](d,stages=4,min_tiles=1)]")
lab=lab.replace("for j,st in enumerate((4,6,8))","for j,st in enumerate(('best128','new2','new3','new4','pair64'))")
lab=lab.replace('FP8 QK + register FP16 PV','N128 one-head raw-V matrix-load PV vs S13 and frozen N1289a23ca9a')
lab=lab.replace('V stays raw, exact register widening; P keeps E4M3 quantization, output register transpose only once per portion.','N128 raw-V one-head CTA160, shared P key permutation, native FP8 PV; stages2/3/4 grid3xSM; comparator best128 uses true native onHopper and S13 ABI stand-in locally.')
(H/'ldsv128_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'vq_runtime.py').read_text(),'best128':bestdrv,'new':drv,'pair64':(H/'ldsvpair.py').read_text()})+'\n'+lab)
