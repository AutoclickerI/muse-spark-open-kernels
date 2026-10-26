"""Correct conditional builds for the P3a operand diagnostic, frozen on P4 S22."""
import ast,base64,difflib,hashlib,importlib.util,json,lzma,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_builds22';O.mkdir(parents=True,exist_ok=True)
sp=importlib.util.spec_from_file_location('p3a_ip',D.parent/'intpipe/build.py');ip=importlib.util.module_from_spec(sp);sp.loader.exec_module(ip)
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
BASE_SHA='4b8c803d'
ARMS=('CACHE1','LAZY','LAZY2','N1SIMT','EPI4')

def expanded():
 p=R/'problems/muse_decode_layer/cand_r3_best7_s22_final.py';snapshot=D/'base_snapshot.json'
 if not snapshot.exists():
  oid=subprocess.check_output(['git','rev-parse','HEAD:'+str(p.relative_to(R))],cwd=R,text=True).strip();snapshot.write_text(json.dumps(dict(shipped=str(p.relative_to(R)),git_blob=oid,shipped_sha256=sha(p.read_text())),indent=2)+'\n')
 meta=json.loads(snapshot.read_text());packed=subprocess.check_output(['git','cat-file','blob',meta['git_blob']],cwd=R,text=True);assert sha(packed)==meta['shipped_sha256'];blob=next(n.args[0].value for n in ast.walk(ast.parse(packed)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');s=lzma.decompress(base64.b85decode(blob)).decode();assert sha(s).startswith(BASE_SHA)
 return s,packed,blob,'problems/muse_decode_layer/dev/s22/generated/s22.py'

def fill_span(cu):
 a=cu.index('#pragma unroll\n',cu.index('probe[16] += tg() - wait0;'));b=cu.index('    }\n    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp)',a)
 return a,b

FILL='''#pragma unroll
      for(int j=0;j<NT;++j){int idx=nb*NB+j*8+(lane>>2);const __nv_bfloat16* row=nullptr;
        if(idx<ne){int pos=off+idx;row=pass?inter+(size_t)pos*HD:x+(size_t)(pair_of[pos]>>3)*HD;}
#pragma unroll
        for(int g=0;g<8;++g){uint4 u=make_uint4(0,0,0,0),v=u;
          if(row){const uint4* p=reinterpret_cast<const uint4*>(row+warp*512+g*64+q*16);if(pass){u=__ldcg(p);v=__ldcg(p+1);}else{u=__ldg(p);v=__ldg(p+1);}}
          *reinterpret_cast<uint4*>(frag+((j*8+g)*2)*NTHR*4+tid*4)=u;
          *reinterpret_cast<uint4*>(frag+((j*8+g)*2+1)*NTHR*4+tid*4)=v;
        }
      }
      __syncwarp();
'''
GROUP='''        uint32_t Bj[NT][8];
#pragma unroll
        for(int j=0;j<NT;++j){uint4 u=make_uint4(0,0,0,0),v=u;
          LOAD
          Bj[j][0]=u.x;Bj[j][1]=u.y;Bj[j][2]=u.z;Bj[j][3]=u.w;Bj[j][4]=v.x;Bj[j][5]=v.y;Bj[j][6]=v.z;Bj[j][7]=v.w;
        }
'''
GLOBAL_LOAD='''int idx=nb*NB+j*8+(lane>>2);if(idx<ne){int pos=off+idx;const __nv_bfloat16* row=pass?inter+(size_t)pos*HD:x+(size_t)(pair_of[pos]>>3)*HD;
          const uint4* p=reinterpret_cast<const uint4*>(row+warp*512+g*64+q*16);if(pass){u=__ldcg(p);v=__ldcg(p+1);}else{u=__ldg(p);v=__ldg(p+1);}}'''
SHARED_LOAD='u=bc_ld4(frag+((j*8+g)*2)*NTHR*4+tid*4);v=bc_ld4(frag+((j*8+g)*2+1)*NTHR*4+tid*4);'
MMA='          for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);'

def transform(cu,arm):
 if arm in ('CACHE1','LAZY','LAZY2','N1SIMT'):
  a,b=fill_span(cu);cu=cu[:a]+(FILL if arm=='CACHE1' else '')+cu[b:]
  assert cu.count('  uint32_t Bf[NT][8][8];')==1;cu=cu.replace('  uint32_t Bf[NT][8][8];','')
  anchor='        const uint4 wv = wq[g * 32];';assert cu.count(anchor)==1
  cu=cu.replace(anchor,GROUP.replace('LOAD',SHARED_LOAD if arm=='CACHE1' else GLOBAL_LOAD)+anchor)
  assert cu.count(MMA)==1
  cu=cu.replace(MMA,MMA.replace('Bf[j][g]','Bj[j]'))
 if arm=='CACHE1':
  start=cu.index('    int M, int local_start, int nstage, uint64_t* __restrict__ trace) {')
  cu=cu[:start]+cu[start:].replace('uint64_t* __restrict__ trace) {','uint64_t* __restrict__ trace) {\n  nstage=1;',1)
  cu=cu.replace('  const int q = lane & 3;','  uint32_t* frag=reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(pair_of+npairs)+127ull)&~127ull);\n  const int q = lane & 3;',1)
  a=cu.index('size_t smem_bytes(');b=cu.index('\n}',a);s=cu[a:b]
  assert 'static_cast<size_t>(nstage)' in s and '2 * nstage * 8' in s
  s=s.replace('static_cast<size_t>(nstage)','static_cast<size_t>(1)').replace('2 * nstage * 8','2 * 1 * 8').replace('4 * npairs;','4 * npairs + 128 + static_cast<size_t>(nt)*65536;');cu=cu[:a]+s+cu[b:]
  # With one raw slot, activation cache begins at the end of unmodified metadata.
  helper='''__device__ __forceinline__ uint4 bc_ld4(const uint32_t* p){uint4 v;asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(smem_u32(p)):"memory");return v;}\n'''
  cu=cu.replace('template <int NT, int F>\n__global__',helper+'template <int NT, int F>\n__global__',1)
 if arm=='LAZY2':
  needle='    float acc[NT][4];';cu=cu.replace(needle,needle+'\n    float acc_b[NT][4] = {};',1)
  call=MMA.replace('Bf[j][g]','Bj[j]');assert cu.count(call)==1;cu=cu.replace(call,call.replace('acc[j]', '(g&1)?acc_b[j]:acc[j]'))
  needle='    __syncwarp();\n    if (lane == 0) mbar_arrive';assert cu.count(needle)==1
  cu=cu.replace(needle,'    for(int j=0;j<NT;++j)for(int z=0;z<4;++z)acc[j][z]+=acc_b[j][z];\n'+needle)
 if arm=='N1SIMT':
  cu=cu.replace('    float acc[NT][4];','    const bool narrow=NT==1&&ne-nb*NB==1;\n    float nx=0.f,ny=0.f;\n    float acc[NT][4];',1)
  call=MMA.replace('Bf[j][g]','Bj[j]');assert cu.count(call)==1
  body='''          if(narrow){
            uint32_t b0=__shfl_sync(~0u,Bj[0][2*s],q),b1=__shfl_sync(~0u,Bj[0][2*s+1],q);
            nx=fmaf(bc_bf(a0,false),bc_bf(b0,false),nx);nx=fmaf(bc_bf(a0,true),bc_bf(b0,true),nx);
            nx=fmaf(bc_bf(a2,false),bc_bf(b1,false),nx);nx=fmaf(bc_bf(a2,true),bc_bf(b1,true),nx);
            ny=fmaf(bc_bf(a1,false),bc_bf(b0,false),ny);ny=fmaf(bc_bf(a1,true),bc_bf(b0,true),ny);
            ny=fmaf(bc_bf(a3,false),bc_bf(b1,false),ny);ny=fmaf(bc_bf(a3,true),bc_bf(b1,true),ny);
          }else{CALL}
'''.replace('CALL',call)
  cu=cu.replace(call,body)
  needle='    __syncwarp();\n    if (lane == 0) mbar_arrive';assert cu.count(needle)==1
  cu=cu.replace(needle,'''    if(narrow){nx+=__shfl_xor_sync(~0u,nx,1);ny+=__shfl_xor_sync(~0u,ny,1);nx+=__shfl_xor_sync(~0u,nx,2);ny+=__shfl_xor_sync(~0u,ny,2);
      acc[0][0]=q==0?nx:0.f;acc[0][2]=q==0?ny:0.f;acc[0][1]=acc[0][3]=0.f;}
'''+needle)
  cu=cu.replace('template <int NT, int F>\n__global__','__device__ __forceinline__ float bc_bf(uint32_t x,bool hi){return __uint_as_float(hi?(x&0xffff0000u):(x<<16));}\n'+'template <int NT, int F>\n__global__',1)
 if arm=='EPI4':
  if 'constexpr int SHARDS' in cu:
   # M1's SHARD2/NREG word formerly excluded vector reduction. Keep SHARD2 but use
   # the same vector intermediate layout and scalar-per-warp norm ownership below.
   old='constexpr bool VEC=3>0&&NT==1&&!NREG,VNR=3==3&&NT==1&&!NREG;';assert old in cu
   cu=cu.replace(old,'constexpr bool VEC=NT==1,VNR=NT==1;').replace('if constexpr (NREG)', 'if constexpr (NREG&&!VEC)')
   cu=cu.replace('  int cur_pass =','  int cur_off=0,cur_ne=0;\n  int cur_pass =',1).replace('cur_nb = nb; seg_tiles = 0;','cur_nb = nb;cur_off=off;cur_ne=ne;seg_tiles = 0;',1)
   # SLOT does not initialize expert-wide n/off metadata. Preserve the passed segment values.
   cu=cu.replace('ne=rt->n[cur_e],off=rt->off[cur_e]','ne=cur_ne,off=cur_off',1)
  a=cu.index('    if constexpr(VEC){',cu.index('    const float* rb = red +'));brace=cu.index('{',a);b=brace+1;depth=1
  while depth:depth+=(cu[b]=='{')-(cu[b]=='}');b+=1
  zero='for(int sh=0;sh<SHARDS;++sh)ss[(off+idx)*SHARDS+sh]=0.f;' if 'constexpr int SHARDS' in cu else 'ss[off+idx]=0.f;'
  scale='' if 'float v0=a,v1=b,z0=' in cu else 'v*=gs2[e]*256.f;'
  body='''    if constexpr(VEC){
      if(!pass){if(warp<4&&lane<16){int i=lane>>1,n=(lane&1)*4+warp,idx=nb*8+n,ol=i*4+(n>>1);float a=0.f,b=0.f;
#pragma unroll
        for(int w=0;w<NCW;++w){float2 v=*reinterpret_cast<const float2*>(rb+w*128+(n&1)*64+ol*2);a+=v.x;b+=v.y;}
        float scale=gs13[e]*256.f,g=a*scale,u=b*scale;if(idx<ne){if constexpr(Z)if(m==0&&i==0){ZERO}
          inter[(size_t)(off+idx)*HD+m*8+i]=__float2bfloat16(g/(1.f+__expf(-g))*u);}
      }}else if(warp<4){int i=(lane>>2)+(warp>>1)*8,n=(lane&3)*2+(warp&1),idx=nb*8+n;float v=0.f;
#pragma unroll
        for(int w=0;w<NCW;++w)v+=rb[w*128+(warp&1)*64+lane*2+(warp>>1)];
        SCALE
        float z=v*(1.f+__bfloat162float(pnw[m*16+i]));if(idx<ne)y[(size_t)(off+idx)*HD+m*16+i]=ZOUT?z:v;
        nv0+=idx<ne?z*z:0.f;
      }
    }'''.replace('ZERO',zero).replace('SCALE',scale)
  cu=cu[:a]+body+cu[b:]
  a=cu.index('  auto flush_vec_norm=');b=cu.index('  auto unit_body',a);part=cu[a:b];assert 'if(warp<2)' in part and '(lane&3)*2+warp' in part
  part=part.replace('if(warp<2)','if(warp<4)').replace('(lane&3)*2+warp','(lane&3)*2+(warp&1)');cu=cu[:a]+part+cu[b:]
  duplicate='  if constexpr(VNR)if(cur_pass==1)flush_vec_norm();\n      if constexpr(VNR)if(cur_pass==1)flush_vec_norm();';assert cu.count(duplicate)==1;cu=cu.replace(duplicate,'      if constexpr(VNR)if(cur_pass==1)flush_vec_norm();')
 return cu

def main():
 base,packed,blob,basepath=expanded();(O/'s22.py').write_text(base);cells={};manifest={}
 for arm in ARMS:
  candidate=base;variants=[]
  for i,prefix in enumerate(ip.PREFIXES):
   cell=ip.cell(base,prefix);node=next(n for n in ast.parse(cell).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'));old=ast.literal_eval(node.value);cu=transform(old,arm);tag='b22_'+arm.lower()+str(i);text=cell.replace(repr(old),repr(cu),1).replace(prefix,tag).replace(prefix.upper(),tag.upper());cp=D/(arm.lower()+str(i)+'_cell.py');cp.write_text(text);(O/(arm.lower()+str(i)+'.cu')).write_text(cu)
   spec=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(text),cuda_sha256=sha(cu),prefix=tag,arm=arm,M=[1] if i==0 else [8] if i==1 else [16,32]);cells[arm+str(i)]=spec;variants.append(spec)
   n=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==prefix.upper()+'_CUDA');candidate=candidate.replace(ast.get_source_segment(base,n),n.targets[0].id+'='+repr(cu),1)
  compile(candidate,'<'+arm+'>','exec');p=O/(arm.lower()+'.py');p.write_text(candidate);spacked=packed.replace(blob,base64.b85encode(lzma.compress(candidate.encode(),preset=9)).decode(),1).replace(sha(base),sha(candidate));ship=O/(arm.lower()+'_packed.py');ship.write_text(spacked);pp=D/(arm.lower()+'.patch');pp.write_text(''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+basepath,tofile='b/'+basepath)))
  manifest[arm]=dict(base=basepath,base_sha256=sha(base),candidate=str(p.relative_to(R)),candidate_sha256=sha(candidate),shipped=str(ship.relative_to(R)),shipped_sha256=sha(spacked),patch=str(pp.relative_to(R)),patch_sha256=sha(pp.read_text()),variants=variants)
 (D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('BUILD_S22',sha(base),list(manifest),flush=True)
if __name__=='__main__':main()
