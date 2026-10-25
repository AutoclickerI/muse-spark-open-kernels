"""Full-q8 occupancy rebuild: one KV head, 128 compute +32 gather threads, three CTA target."""
import ast
import re
import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'regv.cu').read_text()
# One head per stream-K segment and per CTA. Preserve 128-thread WGMMA groups.
s=s.replace('(int)(HK / 2), (int)(B * (HK / 2))','(int)HK, (int)(B * HK)')
s=s.replace('2 * hp + hh','hp')
s=re.sub(r'(?:const )?int hh\s*=\s*__shfl_sync\(0xffffffffu,\s*threadIdx.x\s*>>\s*7,\s*0\);','const int hh=0;',s)
s=s.replace('sm+CLKPAD+24576+hh*4096','sm+CLKPAD+12288')
s=s.replace('sm+CLKPAD+32768+hh*256','sm+CLKPAD+16384')
s=s.replace('(i/1024)*8192','(i/1024)*4096')
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
s=s.replace('__launch_bounds__(384, 1)','__launch_bounds__(160, 3)')
s=s.replace('uint8_t* Kn = Qs + 8192;','uint8_t* Kn = Qs + 4096;')
s=s.replace('uint8_t* VTn = Qs + 16384;','uint8_t* VTn = Qs + 8192;')
s=s.replace('uint8_t* Ks = Qs + 33792;','uint8_t* Ks = Qs + 17408;')
s=s.replace('uint8_t* VR = Ks + ST * 8192;','uint8_t* VR = Ks + ST * 4096;')
s=s.replace('mb_init(&full[tid], 4); mb_init(&empty[tid], 8);','mb_init(&full[tid], 1); mb_init(&empty[tid], 4);')
s=s.replace('if (warp >= 8)', 'if (warp >= 4)')
s=s.replace('CLKPAD + 33792 + ST * 16384','CLKPAD + 17408 + ST * 8192')
s=s.replace('cfg.blockDim = dim3(384);','cfg.blockDim = dim3(160);')
s=s.replace('vq8_main<ST>, 384, smem','vq8_main<ST>, 160, smem')
s=s.replace('dim3(2 * p.L * p.wpr, p.NS)','dim3(p.L * p.wpr, p.NS)')
s=s.replace('sm+33792','sm+17408').replace('<<<1,128,37888,','<<<1,128,21504,')
# Full verify and BF16 input are compile-time facts in this kernel. Projection
# partials are summed and rounded at the same BF16 boundary in the adapter.
a=s.index('template <int ST> constexpr int smem_bytes()')
device=s[:a].replace('p.L','8').replace('p.window','0').replace('p.proj_splits','0')
s=device+s[a:]
old='if (HK % 2 || HQ != 8 * HK || p.NS > NSMAX || (L != 1 && L != 8))'
assert old in s;s=s.replace(old,'if (HQ != 8 * HK || p.NS > NSMAX || L != 8 || window || proj_splits)')
(H/'onehead.cu').write_text(s)
tree=ast.parse((H/'regv.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
 if isinstance(n,ast.FunctionDef) and n.name=='vq8_attention':
  body=[]
  for q in n.body:
   if isinstance(q,ast.If) and 'get_device_capability' in ast.unparse(q.test):continue
   if isinstance(q,ast.Assign) and getattr(q.targets[0],'id','')=='G':q.value=ast.parse('grid or (3*_num_sms(device))',mode='eval').body
   if isinstance(q,ast.Assign) and getattr(q.targets[0],'id','')=='NS':q.value=ast.parse('B*HK',mode='eval').body
   body.append(q)
   if isinstance(q,ast.Assign) and getattr(q.targets[0],'id','')=='L':
    body+=ast.parse("assert L==8 and cfg['window']==0 and clocks is None\nif proj_splits:\n    raw=raw.sum(0).to(torch.bfloat16)\n    proj_splits=0\n    split_stride=0\nassert raw.dtype==torch.bfloat16").body
  n.body=body
drv=ast.unparse(tree).replace('(G + NS) * 128 * 68','(G + NS) * 64 * 68')+'\n'
(H/'onehead.py').write_text(drv)
test=(H/'check_regv.py').read_text().replace("load('regv')","load('onehead')").replace("'regv.py'","'onehead.py'").replace("'regv_validation.json'","'onehead_validation.json'")
(H/'check_onehead.py').write_text(test)
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
lab=(H/'hybrid_lab.py.inc').read_text().replace('FP8 QK + register FP16 PV','one-head CTA160 native FP8 register-V PV').replace('V stays raw, exact register widening; P keeps E4M3 quantization, output register transpose only once per portion.','One KV head/segment,128 compute+32 gather threads, full L8/no-RoPE/BF16 input specialization; grid3xSM, target3CTAs/SM; FP32 partials externally rounded to BF16.')
(H/'onehead_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'vq_runtime.py').read_text(),'new':drv})+'\n'+lab)
print('One head/CTA160:',hashlib.sha256(s.encode()).hexdigest())
