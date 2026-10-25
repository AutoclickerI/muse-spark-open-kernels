"""Occupancy control: one KV head/CTA, incumbent P-register PV and V byte transpose."""
import ast
import hashlib
import re
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'vq8_base.cu').read_text()
s=s.replace('(int)(HK / 2), (int)(B * (HK / 2))','(int)HK, (int)(B * HK)')
s=s.replace('2 * hp + hh','hp')
s=re.sub(r'(?:const )?int hh\s*=\s*__shfl_sync\(0xffffffffu,\s*threadIdx.x\s*>>\s*7,\s*0\);','const int hh=0;',s)
s=s.replace('* 128 * PR','* 64 * PR').replace('slot * 128 +','slot * 64 +').replace('pb * 128 +','pb * 64 +')
s=s.replace('slot * 8192','slot * 4096').replace('% ST) * 8192','% ST) * 4096')
s=s.replace('(int64_t)hp * 128 + c * 16','(int64_t)hp * 64 + c * 16')
s=s.replace('(threadIdx.x >> 5) - 8','(threadIdx.x >> 5) - 4')
s=s.replace('lane >> 3, c = lane & 7','lane >> 2, c = lane & 3')
s=s.replace('const int head = c >> 2, cb = (c & 3) * 16;','const int head = 0, cb = c * 16;')
old='''#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int kk = 16 * pw + 4 * i + q;'''
assert s.count(old)==1;s=s.replace(old,'''#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int kk = 8 * i + q;''')
a=s.index('  const int key = min(key0 + 16 * pw');b=s.index('\n}',a)
s=s[:a]+'''#pragma unroll
  for(int j=0;j<2;++j){
    const int key=min(key0+32*j+lane,max(seq-1,0));
    cp4(ring+slot*64+32*j+lane,p.pt+(int64_t)b*p.pages+(key>>p.lg));
  }'''+s[b:]
s=s.replace('for (int h = 0; h < 2; h++) {','for (int h = 0; h < 1; h++) {')
s=s.replace('transpose_group(VR + slot * 4096, VT + slot * 4096, pw, lane);',
'''#pragma unroll
      for(int group=0;group<4;++group) transpose_group(VR+slot*4096,VT+slot*4096,group,lane);''')
s=s.replace('__launch_bounds__(384, 1)','__launch_bounds__(160, 3)')
s=s.replace('uint8_t* Kn = Qs + 8192;','uint8_t* Kn = Qs + 4096;')
s=s.replace('uint8_t* VTn = Qs + 16384;','uint8_t* VTn = Qs + 8192;')
s=s.replace('uint8_t* Ks = Qs + 24576;','uint8_t* Ks = Qs + 12288;')
s=s.replace('ST * 8192','ST * 4096')
s=s.replace('mb_init(&full[tid], 4); mb_init(&empty[tid], 8);','mb_init(&full[tid], 1); mb_init(&empty[tid], 4);')
s=s.replace('if (warp >= 8)', 'if (warp >= 4)')
s=s.replace('CLKPAD + 24576 + ST * 24576','CLKPAD + 12288 + ST * 12288')
s=s.replace('cfg.blockDim = dim3(384);','cfg.blockDim = dim3(160);')
s=s.replace('vq8_main<ST>, 384, smem','vq8_main<ST>, 160, smem')
s=s.replace('dim3(2 * p.L * p.wpr, p.NS)','dim3(p.L * p.wpr, p.NS)')
ca=s.index('  float2 gate = make_float2',s.index('vq8_combine(VP p)'));cb=s.index("  // This warp's slice",ca)
gate=s[ca:cb];s=s[:ca]+s[cb:]
mark='  *(__nv_bfloat162*)(p.out + ((int64_t)(b * p.L + tok) * p.HQ + head) * 64 + 2 * lane) ='
assert s.count(mark)==1;s=s.replace(mark,gate+mark,1)
a=s.index('template <int ST> constexpr int smem_bytes()')
s=s[:a].replace('p.L','8').replace('p.window','0').replace('p.proj_splits','0')+s[a:]
old='if (HK % 2 || HQ != 8 * HK || p.NS > NSMAX || (L != 1 && L != 8))'
assert old in s;s=s.replace(old,'if (HQ != 8 * HK || p.NS > NSMAX || L != 8 || window || proj_splits)')
(H/'onep.cu').write_text(s)
tree=ast.parse((H/'onehead.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CPP':n.value=ast.Constant(ast.literal_eval(n.value).replace('\nvoid regv_pv_probe(int64_t,int64_t,int64_t,int64_t);\n',''))
drv=ast.unparse(tree).replace(", 'regv_pv_probe'",'')+'\n'
mark='    NS = B * HK\n'
assert drv.count(mark)==1
drv=drv.replace(mark,mark+'''    # Arithmetic/staging stand-in only: sm120 has 99KiB per block, not Hopper's 227KiB.
    # Native ST8 is separately compiled and checked; local ST8 exercises ST6's same ABI.
    if torch.cuda.get_device_capability(device)[0] != 9 and stages == 8:
        stages = 6
''')
(H/'onep.py').write_text(drv)
test=(H/'check_onehead.py').read_text().replace("load('onehead')","load('onep')").replace("'onehead.py'","'onep.py'").replace("'onehead_validation.json'","'onep_validation.json'")
a=test.index('gen=torch.Generator');b=test.index("s=(H/'phases_lab.py')",a)
test=test[:a]+'pv=[]\n'+test[b:]
(H/'check_onep.py').write_text(test)
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
lab=(H/'hybrid_lab.py.inc').read_text().replace('FP8 QK + register FP16 PV','one-head CTA160 incumbent FP8 P-register PV')
lab=lab.replace('V stays raw, exact register widening; P keeps E4M3 quantization, output register transpose only once per portion.','One KV head,128 compute+32 gather threads, one warp transposes four16-key groups; full L8/BF16 specialization; grid3xSM.')
(H/'onep_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'vq_runtime.py').read_text(),'new':drv})+'\n'+lab)
print('P-register occupancy control:',hashlib.sha256(s.encode()).hexdigest())
