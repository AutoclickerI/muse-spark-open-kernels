"""Two writer warps with contiguous float2 planes: keep independent epilogue warps and remove gather conflicts."""
import hashlib
from pathlib import Path
from build_epilogue import source as single_source,VEC
DEV=Path(__file__).resolve().parent
PAIR=r'''
      if(warp<2){float a=0.f,b=0.f;
#pragma unroll
        for(int w=0;w<NCW;++w){float2 v=*reinterpret_cast<const float2*>(rb+w*128+warp*64+lane*2);a+=v.x;b+=v.y;}
        int i=lane>>2,n=(lane&3)*2+warp,idx=nb*8+n;
        if(!pass){float scale=gs13[e]*256.f,g=a*scale,u=b*scale;
          if(idx<ne){if constexpr(Z)if(m==0&&i==0)ss[off+idx]=0.f;
            inter[(size_t)(off+idx)*HD+m*8+i]=__float2bfloat16(g/(1.f+__expf(-g))*u);}
        }else{float scale=gs2[e]*256.f,p0=1.f+__bfloat162float(pnw[m*16+i]),p1=1.f+__bfloat162float(pnw[m*16+i+8]);
          float v0=a*scale,v1=b*scale,z0=v0*p0,z1=v1*p1;
          if(idx<ne){y[(size_t)(off+idx)*HD+m*16+i]=ZOUT?z0:v0;y[(size_t)(off+idx)*HD+m*16+i+8]=ZOUT?z1:v1;}
          nv0+=idx<ne?z0*z0+z1*z1:0.f;
        }
      }
'''
def source():
 cell,cuda=single_source(3);old=cuda
 assert VEC in cuda;cuda=cuda.replace(VEC,PAIR)
 a=cuda.index('  auto flush_vec_norm=');b=cuda.index('  auto unit_body =',a)
 cuda=cuda[:a]+r'''  auto flush_vec_norm=[&](){
    if constexpr(VNR){uint64_t t=0;if constexpr(TRACE)if(tid==0)t=tg();
      if(warp<2){float s=nv0;
#pragma unroll
        for(int d=4;d<=16;d*=2)s+=__shfl_xor_sync(~0u,s,d);
        if(lane<4){int n=cur_nb*8+(lane&3)*2+warp,ne=rt->n[cur_e],off=rt->off[cur_e];if(n<ne)atomicAdd(ss+off+n,s);}
      }
      nv0=0.f;nv1=0.f;if constexpr(TRACE)if(tid==0)probe[60]+=tg()-t;
    }
  };
'''+cuda[b:]
 oldstore='''#pragma unroll
    for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);'''
 assert oldstore in cuda
 newstore='''    if constexpr(VEC){float* p=red+(buf*NCW+warp)*128;
      *reinterpret_cast<float2*>(p+lane*2)=make_float2(acc[0][0],acc[0][2]);
      *reinterpret_cast<float2*>(p+64+lane*2)=make_float2(acc[0][1],acc[0][3]);
    }else{
'''+oldstore+'''\n    }'''
 cuda=cuda.replace(oldstore,newstore)
 cell=cell.replace(repr(old),repr(cuda),1).replace('epin','epit').replace('EPIN','EPIT')
 return cell,cuda
if __name__=='__main__':
 cell,cuda=source()
 for p,s in ((DEV/'epit_cell.py',cell),(DEV/'epit.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
