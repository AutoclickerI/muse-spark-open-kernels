"""Group 2/4 K64 transactions under one barrier and WGMMA commit."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
INSTANCES=[(bt,1,wg,st,mc) for bt,wg,st in
           ((32,1,2),(64,1,2),(64,1,3),(128,1,2),(128,1,3),(64,2,2),(256,1,2))
           for mc in (1,2)]


def source():
    global INSTANCES
    spec=importlib.util.spec_from_file_location('tma_builder',HERE/'build_tma.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    INSTANCES=list(dict.fromkeys(INSTANCES+b.INSTANCES));b.INSTANCES=INSTANCES
    s=b.source()
    def replace(a,c):
        nonlocal s
        assert a in s,a[:100]
        s=s.replace(a,c)
    replace('int nkb, int kx, int pipe, int pdl) {','int nkb, int kx, int pipe, int pdl, int kgroup) {')
    replace('constexpr int STAGE_BYTES = A_BYTES + B_BYTES;',
            'constexpr int SLICE_BYTES = A_BYTES + B_BYTES;\n  const int STAGE_BYTES = SLICE_BYTES * kgroup;')
    replace('const int nsteps = kb1 - kb0;','const int nsteps = (kb1 - kb0) / kgroup;')
    replace('''        unsigned char* sa = smem + stage * STAGE_BYTES;
        mbar_arrive(&full[stage]);
        const int kc = (kb0 + s) * 64;''','''        mbar_arrive(&full[stage]);
        for (int kg = 0; kg < kgroup; ++kg) {
        unsigned char* sa = smem + stage * STAGE_BYTES + kg * SLICE_BYTES;
        const int kc = (kb0 + s * kgroup + kg) * 64;''')
    replace('''                        tok0 + part * (BT / MC));
#endif
        }
      }
    }
  } else''','''                        tok0 + part * (BT / MC));
#endif
        }
        }  // K transaction group
      }
    }
  } else''')
    start=s.index('#ifdef MP_SM90A',s.index('mbar_wait(&full[stage], ph);'))
    end=s.index('#endif',start)+len('#endif')
    old=s[start:end];mma,emu=old.split('#else',1)
    mma=mma.removeprefix('#ifdef MP_SM90A\n')
    op,commit=mma.split('      wgmma_commit();',1)
    fence='''#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);
      wgmma_fence();'''
    assert fence in op
    op=op.replace(fence,'')
    emu=emu.removesuffix('#endif')
    region='''#ifdef MP_SM90A
'''+fence+'''
      for (int kg = 0; kg < kgroup; ++kg) {
        unsigned char* sa = stage_base + kg * SLICE_BYTES;
'''+op+'''
      }
      wgmma_commit();'''+commit+'''#else
      for (int kg = 0; kg < kgroup; ++kg) {
        unsigned char* sa = stage_base + kg * SLICE_BYTES;
'''+emu+'''
      }
#endif'''
    s=s[:start]+region+s[end:]
    replace('''      unsigned char* sa = smem + stage * STAGE_BYTES;
#ifdef MP_SM90A''','''      unsigned char* stage_base = smem + stage * STAGE_BYTES;
#ifdef MP_SM90A''')
    replace('torch::Tensor& Out, int N, int ks, int waves, int pipe, int pdl) {',
            'torch::Tensor& Out, int N, int ks, int waves, int pipe, int pdl, int kgroup) {')
    replace('const size_t stage = (size_t)(BR + BT) * 64 * 2;',
            'TORCH_CHECK(kgroup == 1 || kgroup == 2 || kgroup == 4, "K group");\n  TORCH_CHECK((KW/64) % (ks*kgroup) == 0, "K partition/group");\n  const size_t stage = (size_t)(BR + BT) * 64 * 2 * kgroup;')
    replace('KW / 64, KX, pipe, pdl);','KW / 64, KX, pipe, pdl, kgroup);')
    replace('int64_t waves, int64_t pipe, bool pdl) {','int64_t waves, int64_t pipe, bool pdl, int64_t kgroup) {')
    replace('(int)waves, (int)pipe, (int)pdl);','(int)waves, (int)pipe, (int)pdl, (int)kgroup);')
    return s


def build():
    src=source();(HERE/'grouped.cu').write_text(src)
    tail=(HERE/'tma_tail.py').read_text()
    tail=tail.replace('VP_TMA_SOURCE','VG_SOURCE').replace('VP_TMA_INSTANCES','VG_INSTANCES')
    tail=tail.replace('vp_tma','vg_tma').replace('_vp_tma','_vg_tma')
    tail=tail.replace('int64_t,int64_t,bool);','int64_t,int64_t,bool,int64_t);')
    tail=tail.replace("name='muse_verify_tma_'","name='muse_verify_grouped_'")
    tail=tail.replace('bt,nsub,ncw,st,mc,ks,waves,pipe=choice','bt,nsub,ncw,st,mc,ks,waves,pipe,kgroup=choice')
    tail=tail.replace('st,ks,mc,waves,pipe,pdl)','st,ks,mc,waves,pipe,pdl,kgroup)')
    # K-group ordering is interleaved across tile families, then split/grid.
    tail=tail[:tail.index('\ndef vg_tma_choices(')]+'''
def vg_tma_choices(m,n,k,limit=227328):
    choices=[]
    for kgroup in (2,4,1):
        for bt,ns,wg,st,mc in VG_INSTANCES:
            br=64*ns*wg;shared=max(st*(br+bt)*128*kgroup+16*st+1024,64*(br+4)*4+1024)
            if bt>m*2 or n%br or (n//br)%mc or shared>limit:continue
            for ks in (1,2,4):
                if ks*mc>8 or (k//64)%(ks*kgroup):continue
                for waves in (0,1,2):choices.append((bt,ns,wg,st,mc,ks,waves,1,kgroup))
    return choices
'''
    path=HERE/'grouped_projection.py'
    path.write_text((HERE/'library.py').read_text()+'\nVG_SOURCE='+repr(src)+'\nVG_INSTANCES='+repr(INSTANCES)+'\n'+tail)
    return path


if __name__=='__main__':print(build())
