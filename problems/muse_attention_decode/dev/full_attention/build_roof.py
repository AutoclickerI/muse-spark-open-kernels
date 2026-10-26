"""Full-attention topology and transport arms, each on exact S15."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];out=H/'roof';out.mkdir(exist_ok=True)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest().startswith('60254883')
def value(key):return ast.literal_eval(next(n.value for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key))
def edit(c,a,b):assert c.count(a)==1,(a[:70],c.count(a));return c.replace(a,b)
def assign(s,name,c):
 n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name);lines=s.splitlines(True);lines[n.lineno-1:n.end_lineno]=[name+' = '+repr(c)+'\n'];return ''.join(lines)

def head(grid):
 c=value('VQ8_CUDA')
 c='// P5B_ONE_HEAD: one producer WG + one consumer WG per CTA.\n'+c
 c=edit(c,'head = c >> 2, cb = (c & 3) * 16','head = 0, cb = c * 16')
 c=edit(c,'(threadIdx.x >> 5) - 8','(threadIdx.x >> 5) - 4')
 c=edit(c,'q = lane >> 3, c = lane & 7','q = lane >> 2, c = lane & 3')
 c=edit(c,'const int64_t hoff = (int64_t)hp * 128 + c * 16;','const int64_t hoff = (int64_t)hp * 64 + c * 16;')
 c=edit(c,'for (int i = 0; i < 8; i++) {\n        const int kk = 16 * pw + 4*(i&3) + q + 64*(i>>2);','for (int i = 0; i < 4; i++) {\n        const int kk = 16 * pw + 8*(i&1) + q + 64*(i>>1);')
 # All tile strides halve; per-head 8192 offsets are dead (hh/head=0).
 c=c.replace('slot * 16384','slot * 8192').replace('ST * 16384','ST * 8192')
 c=c.replace('% ST) * 16384','% ST) * 8192')
 c=edit(c,'for (int h = 0; h < 2; h++) {\n#pragma unroll\n    for (int dp','for (int h = 0; h < 1; h++) {\n#pragma unroll\n    for (int dp')
 c=edit(c,'const int hh = __shfl_sync(0xffffffffu, threadIdx.x >> 7, 0);','const int hh = 0;')
 c=edit(c,'const int kvh = 2 * hp + hh;','const int kvh = hp;')
 c=edit(c,'((2 * hp + hh) * 8 + (row & 7))','(hp * 8 + (row & 7))')
 c=c.replace('(int)(HK / 2), (int)(B * (HK / 2))','(int)HK, (int)(B * HK)')
 c=edit(c,'  // A segment has at most grid/NS',f'  p.grid *= {grid};\n  // A segment has at most grid/NS')
 c=edit(c,'const int np_max = (int)(grid / p.NS) + 2;','const int np_max = (int)(p.grid / p.NS) + 2;')
 # Partial slots contain one head's64 rows. Double grid/NS at H2 gives
 # exactly the same allocation as S15's pair-head128-row partial buffer.
 c=c.replace('* 128 * PR','* 64 * PR').replace('* 128 + r','* 64 + r')
 c=edit(c,'slot * 128 + hh * 64','slot * 64 + hh * 64')
 c=edit(c,'mb_init(&empty[tid], 8);','mb_init(&empty[tid], 4);')
 c=edit(c,'const int head = (2 * hp + hh) * 8 + (row & 7);','const int head = hp * 8 + (row & 7);')
 c=edit(c,'cfg.gridDim = dim3(2 * p.L * p.wpr, p.NS);','cfg.gridDim = dim3(p.L * p.wpr, p.NS);')
 c=edit(c,'__launch_bounds__(384, 1) vq8_main','__launch_bounds__(256, 2) vq8_main')
 c=edit(c,'cfg.blockDim = dim3(384);','cfg.blockDim = dim3(256);')
 c=edit(c,'&active, vq8_main<ST>, 384, smem','&active, vq8_main<ST>, 256, smem')
 c=edit(c,'  if (warp >= 8) {','''#if USE_WGMMA
  if(warp>=4)asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 216;" ::: "memory");
#endif
  if (warp >= 4) {''')
 c=edit(c,'uint8_t* Kn = Qs + 8192;\n  uint8_t* VTn = Qs + 16384;\n  uint8_t* Ks = Qs + 24576;',
 'uint8_t* Kn = Qs + 4096;\n  uint8_t* VTn = Qs + 8192;\n  uint8_t* Ks = Qs + 12288;')
 c=edit(c,'return CLKPAD + 24576 + ST * 49152;','return CLKPAD + 12288 + ST * 24576;')
 # The only non-Hopper N128 fallback bug: WG-local Q indexing.
 c=edit(c,'w=threadIdx.x>>5;','w=(threadIdx.x>>5)&3;')
 # Clock roles follow the new producer WG; the old WG1-consumer stamp is dead.
 c=c.replace('CLK_ADD(256,','CLK_ADD(128,').replace('if (tid == 256)', 'if (tid == 128)')
 c=c.replace('row[7] = a[8]; row[8] = a[32];','row[7] = a[4]; row[8] = a[28];')
 c=c.replace('  if (tid == 128) { row[11] = clk_area()[4]; row[13] = gtime(); }','')
 return c

def balanced():
 c=value('PARTIAL_CUDA')
 old='  const int c_lo = min(lo + split * chunk, seq);\n  const int c_hi = min(c_lo + chunk, seq);'
 new='''  int c_lo = min(lo + split * chunk, seq),c_hi = min(c_lo + chunk, seq);
  if constexpr(L==1){if(!p.window){
    const int tiles=(n_cached+T-1)/T;
    c_lo=min(lo+(split*tiles/p.ns)*T,seq);
    c_hi=min(lo+((split+1)*tiles/p.ns)*T,seq);
  }}'''
 return edit(c,old,new)

def gather():
 c=value('ATTN_PORT_CUDA_SRC')
 old='''    const int t = c_lo + tile * T + lt + 16 * k;
      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;'''
 # Re-map each quad's4 rows to adjacent keys, vectorizing the page-table load.
 a=c.index('  auto load_rows =');b=c.index('  auto issue =',a)
 c=c[:a]+'''  auto load_rows = [&](int tile, int (&rows)[4]) {
    if constexpr(L==1){if(!p.window && p.ps==1){
      const int at=c_lo+tile*T+4*lt;
      if(tile<ntiles && at+3<c_hi && at+7<p.max_pages){
        const uintptr_t addr=reinterpret_cast<uintptr_t>(pt+at),aligned=addr&~uintptr_t(15);
        const int shift=(addr>>2)&3;const int4 z=*(const int4*)aligned;int4 y;
        if(shift)y=*(const int4*)(aligned+16);
        if(shift==0){rows[0]=z.x;rows[1]=z.y;rows[2]=z.z;rows[3]=z.w;}
        else if(shift==1){rows[0]=z.y;rows[1]=z.z;rows[2]=z.w;rows[3]=y.x;}
        else if(shift==2){rows[0]=z.z;rows[1]=z.w;rows[2]=y.x;rows[3]=y.y;}
        else{rows[0]=z.w;rows[1]=y.x;rows[2]=y.y;rows[3]=y.z;}
      }else{for(int k=0;k<4;k++)rows[k]=(tile<ntiles && at+k<c_hi)?pt[at+k]:-1;}
      return;
    }}
#pragma unroll
    for(int k=0;k<4;k++){
      const int t=c_lo+tile*T+lt+16*k;
      rows[k]=(tile<ntiles && t<c_hi)?(pt[t>>p.ps_log2]*p.ps+(t&(p.ps-1))):-1;
    }
  };
'''+c[b:]
 c=edit(c,'      const int t = lt + 16 * k;','      const int t = (L==1 && !p.window && p.ps==1)?4*lt+k:lt+16*k;')
 return c

def main():
 meta={}
 for name,key,c in [('H1','VQ8_CUDA',head(1)),('H2','VQ8_CUDA',head(2)),('F','PARTIAL_CUDA',balanced()),('G','ATTN_PORT_CUDA_SRC',gather()),('P',None,None)]:
  s=assign(base,key,c) if key else base
  if key=='VQ8_CUDA':s=s.replace('muse_vq8_9a23ca9a98dd_','muse_vq8_'+hashlib.sha256(c.encode()).hexdigest()[:12]+'_')
  if key=='PARTIAL_CUDA':s=s.replace('PARTIAL_FAST_CUDA = PARTIAL_CUDA','PARTIAL_FAST_CUDA = PARTIAL_CUDA')
  if name=='P':s=edit(s,"_PP_ROWS = ((1, 1, False),)","_PP_ROWS = ((1, 1, False),(1,8,False))")
  if name.startswith('H'):
   s=edit(s,'_VQ8_ROWS = {(16, 8, False): (4, 8), (32, 8, False): (4, 8)}','_VQ8_ROWS = {(1,8,False):(4,2),(16,8,False):(4,8),(32,8,False):(4,8)}')
  (out/(name+'.py')).write_text(s)
  if c:(out/(name+'.cu')).write_text(c)
  patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'));(out/(name+'.patch')).write_text(patch)
  meta[name]=dict(source_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(c.encode()).hexdigest() if c else None,patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
 (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
if __name__=='__main__':main()
