"""Unsent responses to the pipe/latency lab: flight, independent MMA chains, K16/r64 and FP4-only LUT."""
import hashlib
from pathlib import Path
from build_epilogue import source as base_source
DEV=Path(__file__).resolve().parent
KINDS={'flight':'pfly','ilp':'pilp','k16':'pk16','lut':'plut'}
PACK=r'''
import triton
import triton.language as tl
@triton.jit
def plut_pack_kernel(P,Q,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);tile=i//2048;k=i%2048;base=tile*9216+k*4
 a=tl.load(P+base,i<N,0);b=tl.load(P+base+1,i<N,0);c=tl.load(P+base+2,i<N,0);d=tl.load(P+base+3,i<N,0)
 for half in tl.static_range(2):
  A=tl.where(half==0,a,c);Bv=tl.where(half==0,b,d);W=(A&0x60306030)|(Bv&0x9fcf9fcf)
  p0=tl.full((B,),0,tl.uint32);p1=tl.full((B,),0,tl.uint32)
  for j in tl.static_range(8):
   if j==0:v=A
   elif j==1:v=(A>>4)|(A<<28)
   elif j==2:v=(A>>10)|(A<<22)
   elif j==3:v=(Bv>>2)|(Bv<<30)
   elif j==4:v=(Bv>>6)|(Bv<<26)
   elif j==5:v=(Bv>>12)|(Bv<<20)
   elif j==6:v=(W>>24)|(W<<8)
   else:v=(W>>30)|(W<<2)
   q=((v>>6)&7)|((v>>12)&8)|((v>>18)&112)|((v>>24)&128)
   if j<4:p0|=q<<(j*8)
   else:p1|=q<<((j-4)*8)
  tl.store(Q+base+half*2,p0,i<N);tl.store(Q+base+half*2+1,p1,i<N)
def plut_pack(t):
 out=t.clone();n=t.numel()//36864*2048
 plut_pack_kernel[(triton.cdiv(n,256),)](t.view(torch.uint32),out.view(torch.uint32),n,256)
 return out
'''

def source(kind):
 cell,cuda=base_source(0);old=cuda
 if kind=='flight':
  a='else prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);'
  b='''else {prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);
        if(rt->pre.index<total_count){const auto& p=rt->pre;const uint8_t* src=(p.pass?w2p:w13p)+(static_cast<size_t>(p.e)*(p.pass?NT2:NT1)+p.m)*TILE_BYTES;
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"::"l"(src),"r"(TILE_BYTES):"memory");}}
'''
  assert a in cuda;cuda=cuda.replace(a,b)
 elif kind=='ilp':
  anchor='    float acc[NT][4];';assert anchor in cuda;cuda=cuda.replace(anchor,anchor+' float alt[NT][4] = {};')
  anchor='mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);'
  assert anchor in cuda;cuda=cuda.replace(anchor,'mma16816((g&1)?alt[j]:acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);')
  anchor='    __syncwarp();\n    if (lane == 0) mbar_arrive(&emptyb[st]);';assert anchor in cuda
  cuda=cuda.replace(anchor,'''#pragma unroll
    for(int j=0;j<NT;++j){
#pragma unroll
      for(int k=0;k<4;++k)acc[j][k]+=alt[j][k];}
'''+anchor)
 elif kind=='k16':
  cuda=cuda.replace('constexpr int NCW = 8;', 'constexpr int NCW = 16;')
  cuda=cuda.replace('  uint32_t Bf[NT][8][8];','  uint32_t Bf[NT][4][8];')
  # Every warp owns K256 instead of K512. Existing bank halves are interleaved between warp pairs.
  cuda=cuda.replace('for (int g = 0; g < 8; ++g)', 'for (int g = 0; g < 4; ++g)')
  cuda=cuda.replace('warp * 512 + g * 64','warp * 256 + g * 64')
  cuda=cuda.replace('warp * 8 * 32 + lane', '(warp>>1) * 8 * 32 + (warp&1)*4*32 + lane')
  cuda=cuda.replace('warp * 32 + lane', '(warp>>1) * 32 + lane')
  cuda=cuda.replace('scw[g >> 1]', 'scw[(g+((warp&1)*4)) >> 1]')
  # Limit decoded operand live ranges: eight values then consume, rather than sixteen.
  a=cuda.index('        uint32_t o[16];');b=cuda.index('\n      }\n    }\n    __syncwarp();',a)
  cuda=cuda[:a]+'''        uint32_t o[8];
#pragma unroll
        for(int half=0;half<2;++half){
          dq8(half?wv.z:wv.x,half?wv.w:wv.y,o);
#pragma unroll
          for(int s=0;s<2;++s){const uint32_t* os=o+s*4;
            const uint32_t a0=hmul2(os[0],Sr),a1=hmul2(os[1],Sr8),a2=hmul2(os[2],Sr),a3=hmul2(os[3],Sr8);
#pragma unroll
            for(int j=0;j<NT;++j)mma16816(acc[j],a0,a1,a2,a3,Bf[j][g][half*4+2*s],Bf[j][g][half*4+2*s+1]);
          }
        }'''+cuda[b:]
  cuda=cuda.replace('uint32_t Bf[NT][4][8];','uint32_t Bf[NT][3][8];')
  anchor='  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;'
  cuda=cuda.replace(anchor,'  uint32_t* bsave=reinterpret_cast<uint32_t*>(pair_of+npairs);\n'+anchor)
  for k,expr in enumerate(['v0.x','v0.y','v0.z','v0.w','v1.x','v1.y','v1.z','v1.w']):
   a=f'Bf[j][g][{k}] = {expr};';b=f'if(g<3)Bf[j][g][{k}]={expr};else bsave[(j*8+{k})*NTHR+tid]={expr};'
   assert a in cuda;cuda=cuda.replace(a,b)
  for v in ('half*4+2*s','half*4+2*s+1'):
   cuda=cuda.replace(f'Bf[j][g][{v}]',f'(g<3?Bf[j][g][{v}]:bsave[(j*8+{v})*NTHR+tid])')
  a='sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;';assert a in cuda
  cuda=cuda.replace(a,a[:-1]+' + nt*8*NTHR*sizeof(uint32_t);')
  # The combine owns2048 columns per item, hence256 threads even in a512-thread math CTA.
  a="""  for (int it = LAST ? 0 : c; it < 2 * M; it += LAST ? 1 : G) {
    const int t = it >> 1;"""
  assert a in cuda;cuda=cuda.replace(a,a.replace('    const int t','    if(tid<256){\n    const int t'))
  a="""  }
  dump(LAST || c < 2 * M);"""
  assert a in cuda;cuda=cuda.replace(a,'  }\n'+a)
  # Keep the64-slot phase ABI: do not overwrite52..55 with extra warp-exit stamps.
  cuda=cuda.replace('if ((tid & 31) == 0) probe[32 + warp] = tg();','if (tid == 0) probe[32] = tg();')
  cuda=cuda.replace('if (lane == 0) trace[(size_t)c * 64 + 40 + warp] = tg();','if (tid == 0) trace[(size_t)c * 64 + 40] = tg();')
 elif kind=='lut':
  anchor='  const int G = gridDim.x, c = blockIdx.x;'
  assert anchor in cuda;cuda=cuda.replace(anchor,anchor+'''
  __shared__ uint32_t dqtable[256];
  if(tid<256)dqtable[tid]=((tid&7)<<6)|((tid&8)<<12)|((tid&112)<<18)|((tid&128)<<24);
''')
  anchor='''        dq8(wv.x, wv.y, o);
        dq8(wv.z, wv.w, o + 8);'''
  assert anchor in cuda
  cuda=cuda.replace(anchor,'''#pragma unroll
        for(int k=0;k<16;++k){const uint32_t z=k<4?wv.x:k<8?wv.y:k<12?wv.z:wv.w;
          uint32_t ix=__byte_perm(z,0,0x4440u+(k&3));o[k]=dqtable[ix];}
''')
  anchor="    return state\n\n\ndef epib_experts"
  assert anchor in cell;cell=cell.replace(anchor,"    if 'lut' not in state['banks']:\n        state['banks']['lut']={k:plut_pack(v) for k,v in state['banks']['lop3'].items()}\n    return state\n\n\ndef epib_experts")
  cell=cell.replace("bank = state['banks']['lop3']", "bank = state['banks']['lut']")
 else:raise ValueError(kind)
 cell=cell.replace(repr(old),repr(cuda),1)
 tag=KINDS[kind];cell=cell.replace('epib',tag).replace('EPIB',tag.upper())
 if kind=='lut':cell+=PACK
 return cell,cuda

def build():
 for kind,tag in KINDS.items():
  cell,cuda=source(kind)
  for p,s in ((DEV/(tag+'_cell.py'),cell),(DEV/(tag+'.cu'),cuda)):
   p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest()[:8])
if __name__=='__main__':build()
