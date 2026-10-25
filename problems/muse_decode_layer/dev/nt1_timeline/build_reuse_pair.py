"""Fixed ONCE math with two conflict-free writer warps and an independent TMA producer."""
import hashlib
from pathlib import Path
from build_reuse_epilogue import source as old_source
from build_epilogue import VEC
from build_epilogue_pair import PAIR
DEV=Path(__file__).resolve().parent
def source():
 cell,cuda=old_source();old=cuda
 vec=VEC.replace('idx=nb*8+n','idx=j*8+n')
 pair=PAIR.replace('idx=nb*8+n','idx=j*8+n').replace('          nv0+=idx<ne?z0*z0+z1*z1:0.f;', '''          float sq=idx<ne?z0*z0+z1*z1:0.f;
#pragma unroll
          for(int d=4;d<=16;d*=2)sq+=__shfl_xor_sync(~0u,sq,d);
          if(lane<4&&idx<ne)atomicAdd(ss+off+idx,sq);''')
 assert vec in cuda;cuda=cuda.replace(vec,pair)
 a='''       *reinterpret_cast<float4*>(red+((buf*NCW+warp)*32+lane)*4)=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);'''
 b='''       if constexpr(MODE==2){float* p=red+(buf*NCW+warp)*128;
        *reinterpret_cast<float2*>(p+lane*2)=make_float2(acc[j][0],acc[j][2]);
        *reinterpret_cast<float2*>(p+64+lane*2)=make_float2(acc[j][1],acc[j][3]);
       }else{
'''+a+'''\n       }'''
 assert a in cuda;cuda=cuda.replace(a,b)
 cell=cell.replace(repr(old),repr(cuda),1).replace('reusee_','reuseh_').replace('REUSEE_','REUSEH_')
 cuda=cuda.replace('reusee_','reuseh_').replace('REUSEE_','REUSEH_')
 return cell,cuda
if __name__=='__main__':
 cell,cuda=source()
 for p,s in ((DEV/'reuseh_cell.py',cell),(DEV/'reuseh.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
