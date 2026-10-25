"""Round2: shorten K-reduction dependencies, spread writers, and remove global norm atomics."""
import hashlib
from pathlib import Path
from build_epilogue_pair import source as pair_source,PAIR
DEV=Path(__file__).resolve().parent
KINDS={'tree':'epiu','four':'epif','local':'epil','four_local':'epix'}

def source(kind):
 cell,cuda=pair_source();old=cuda
 four=kind in ('four','four_local');local=kind in ('local','four_local');tree=kind=='tree'
 body=PAIR
 if tree:
  a=body.index('float a=0.f,b=0.f;');b=body.index('        int i=',a)
  body=body[:a]+'''float2 r[8];
#pragma unroll
        for(int w=0;w<NCW;++w)r[w]=*reinterpret_cast<const float2*>(rb+w*128+warp*64+lane*2);
        float a=((r[0].x+r[1].x)+(r[2].x+r[3].x))+((r[4].x+r[5].x)+(r[6].x+r[7].x));
        float b=((r[0].y+r[1].y)+(r[2].y+r[3].y))+((r[4].y+r[5].y)+(r[6].y+r[7].y));
'''+body[b:]
 if four:
  body=body.replace('if(warp<2)', 'if(warp<4&&lane<16)').replace('rb+w*128+warp*64+lane*2','rb+w*128+(warp&1)*64+(warp>>1)*32+lane*2').replace('int i=lane>>2,n=(lane&3)*2+warp','int i=(lane>>2)+(warp>>1)*4,n=(lane&3)*2+(warp&1)')
 assert PAIR in cuda;cuda=cuda.replace(PAIR,body)
 a=cuda.index('  auto flush_vec_norm=');b=cuda.index('  auto unit_body =',a)
 writers='warp<4&&lane<16' if four else 'warp<2'
 mask='0x0000ffffu' if four else '~0u';dmax=8 if four else 16
 token='(lane&3)*2+(warp&1)' if four else '(lane&3)*2+warp'
 dest=('norm_local[2*(off+n)'+('+(warp>>1)' if four else '')+']+=s;') if local else 'atomicAdd(ss+off+n,s);'
 cuda=cuda[:a]+f'''  auto flush_vec_norm=[&](){{
    if constexpr(VNR){{uint64_t t=0;if constexpr(TRACE)if(tid==0)t=tg();
      if({writers}){{float s=nv0;
#pragma unroll
        for(int d=4;d<={dmax};d*=2)s+=__shfl_xor_sync({mask},s,d);
        if(lane<4){{int n=cur_nb*8+{token},ne=rt->n[cur_e],off=rt->off[cur_e];if(n<ne){dest}}}
      }}
      nv0=nv1=0.f;if constexpr(TRACE)if(tid==0)probe[60]+=tg()-t;
    }}
  }};
'''+cuda[b:]
 if local:
  cuda=cuda.replace('  constexpr bool PDL =', '  constexpr bool LOCAL=VNR;\n  constexpr bool PDL =')
  anchor='  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;'
  cuda=cuda.replace(anchor,'  float* norm_local=reinterpret_cast<float*>(pair_of+npairs);\n'+anchor)
  cuda=cuda.replace('  if constexpr (PDL) asm volatile', '  if constexpr(LOCAL)for(int i=tid;i<2*npairs;i+=NTHR)norm_local[i]=0.f;\n  if constexpr (PDL) asm volatile',1)
  anchor='  // ---------------- grid barrier (or last-arriver election) ----------------'
  cuda=cuda.replace(anchor,'''  if constexpr(LOCAL){uint64_t t=0;if constexpr(TRACE)if(tid==0)t=tg();
    bar_consumers();
    for(int i=tid;i<npairs;i+=NTHR)ss[npairs+(size_t)i*G+c]=norm_local[2*i]+norm_local[2*i+1];
    if constexpr(TRACE)if(tid==0)probe[61]+=tg()-t;
  }
'''+anchor)
  anchor='    const int t = it >> 1;'
  cuda=cuda.replace(anchor,anchor+'''
    if constexpr(LOCAL){uint64_t tick=0;if constexpr(TRACE)if(tid==0)tick=tg();
      int pos=pos_of[t*8+warp];float sum=0.f;
      if(pos>=0)for(int k=lane;k<G;k+=32)sum+=__ldcg(ss+npairs+(size_t)pos*G+k);
#pragma unroll
      for(int d=16;d>0;d>>=1)sum+=__shfl_xor_sync(~0u,sum,d);
      if(lane==0)norm_local[warp]=pos>=0?tw[t*8+warp]*rsqrtf(sum*(1.f/HD)+1e-8f):0.f;
      bar_consumers();if constexpr(TRACE)if(tid==0)probe[62]+=tg()-tick;
    }
''')
  anchor='''      if constexpr (RSQ) {
        pos = __shfl_sync(~0u, posl, j);'''
  assert anchor in cuda
  cuda=cuda.replace(anchor,'''      if constexpr(LOCAL){pos=pos_of[t*8+j];sc=norm_local[j];}
      else if constexpr (RSQ) {
        pos = __shfl_sync(~0u, posl, j);''')
  anchor='sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;'
  assert anchor in cuda;cuda=cuda.replace(anchor,anchor[:-1]+' + 8 * npairs;')
  # Reuse the existing ss argument; per-replay scratch only, no input-dependent launch or prepare values.
  anchor="ss = torch.empty((p,), device=dev)"
  assert anchor in cell;cell=cell.replace(anchor,"ss = torch.empty((p * (grid + 1),), device=dev)")
 cell=cell.replace(repr(old),repr(cuda),1)
 tag=KINDS[kind];cell=cell.replace('epit',tag).replace('EPIT',tag.upper())
 return cell,cuda

def build():
 for kind,tag in KINDS.items():
  cell,cuda=source(kind)
  for p,s in ((DEV/(tag+'_cell.py'),cell),(DEV/(tag+'.cu'),cuda)):
   p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
