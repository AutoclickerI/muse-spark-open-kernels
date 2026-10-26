"""S18+R8+D: fixed-reference softmax and single-launch full-context attention."""
import ast,difflib,hashlib,importlib.util,json,re,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'fixed';o.mkdir(exist_ok=True)
def one(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a));return s.replace(a,b)
def last():
    # Reuse the existing native gather/MMA, without editing the sliding lane.
    sys.path.insert(0,str(R/'problems/muse_attention_decode/dev/cluster_sliding'))
    import build_last
    c=build_last.source()
    c=one(c,'lo=max(0,seq-p.window+1)','lo=p.window?max(0,seq-p.window+1):0')
    c=one(c,'key>=seq+tok-window+1','(!window || key>=seq+tok-window+1)')
    c=one(c,'if(w==0)csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);','if(w==0)csbuf[lane]=p.window?rope_cs((float)p.positions[b]*p.inv_freq[lane]):make_float2(1.f,0.f);')
    c=one(c,'__sincosf((float)p.positions[b*L+(i>>5)]*p.inv_freq[i&31],&sn,&cs);','if(p.window)__sincosf((float)p.positions[b*L+(i>>5)]*p.inv_freq[i&31],&sn,&cs);else{sn=0.f;cs=1.f;}')
    c=one(c,'const int head=kh*8+sub*H+r;float2 gate=make_float2(0.f,0.f);','const int token=r/H,head=kh*8+sub*H+r%H;float2 gate=make_float2(0.f,0.f);')
    c=one(c,'if(p.gate)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);','''if(p.gate){
      if constexpr(L==1)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);
      else{int tk[1]={b*L+token},cl[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool vd[1]={true};float2 x[1];input_pairs<1>(p,tk,cl,vd,x);gate=x[0];}
    }''')
    c=one(c,'p.out+((int64_t)b*p.HQ+head)*64','p.out+((int64_t)(b*L+token)*p.HQ+head)*64')
    c=one(c,'if(L!=1 || ns<1 || ns>32)throw std::runtime_error("last CTA sliding requires decode and 1..32 splits");','if((L!=1&&L!=8)||ns<1||ns>256)throw std::runtime_error("full last CTA requires L1/8 and 1..256 splits");')
    c=one(c,'if(L==1){if(qt==4){RES(1,4);}else{RES(1,1);}}\n  else throw std::runtime_error("decode only");','if(L==1){RES(1,1);}else{RES(8,1);}')
    c=one(c,'if(L==1){if(qt==4){GO(1,4);}else{GO(1,1);}}','if(L==1){GO(1,1);}else{GO(8,1);}')
    return c

def fixed(c):
    c=one(c,'k_scale is applied on device\n};','k_scale is applied on device\n  const float* fixed;\n};')
    # Fixed reference is a true all-key maximum, scaled exactly as the QK path.
    # m is initialized once per portion; all tile maxima/rescale disappear.
    a=c.index('  float mx[2];\n',c.index('DI void softmax('));b=c.index('    const float bias = 8.f - safe;',a)
    c=c[:a]+'''#pragma unroll
  for(int rh=0;rh<2;rh++){
    const float safe=m[rh];alpha[rh]=1.f;
'''+c[b:]
    a=c.index('DI void rescale(');b=c.index('\n// A portion',a)
    c=c[:a]+'''DI void rescale(float (&O)[9][4],const float (&alpha)[2]) {(void)O;(void)alpha;}
'''+c[b:]
    needle='float O[9][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, alpha[2];'
    c=one(c,needle,'''float O[9][4],m[2],l[2]={},alpha[2];
    for(int rh=0;rh<2;rh++){
      const int row=w*16+g+8*rh,tok=row>>3,head=(2*hp+hh)*8+(row&7);
      m[rh]=live?p.fixed[(b*p.L+tok)*p.HQ+head]:0.f;
    }''')
    c=one(c,'int64_t split_stride, int64_t flags, int64_t stream) {','int64_t split_stride, int64_t flags, int64_t stream,int64_t fixed) {')
    c=one(c,'  // A segment has at most grid/NS','  p.fixed=(const float*)fixed;\n  // A segment has at most grid/NS')
    return c

def bound(c):
    # A K-only pass over the same scheduler/QK layouts. No V history is loaded.
    c=one(c,'        cp16(vd + off(kk, cb), p.vc + src, valid);','')
    c=one(c,'      transpose_group(VR + slot * 16384, VT + slot * 16384, pw, lane);','')
    at=c.index('    float O[9][4], m[2] =');end=c.index('\n}\n\ntemplate <int ST>\n__global__',at)
    c=c[:at]+'''    float mx[2]={-INFINITY,-INFINITY};
    auto reduce=[&](auto& sx,int key0,int keys){
      constexpr int NB=sizeof(sx)/sizeof(sx[0]);
      for(int rh=0;rh<2;rh++){
        const int qr=w*16+g+8*rh,tok=qr>>3;
        float v=mx[rh];
        #pragma unroll
        for(int n=0;n<NB;n++)for(int e=0;e<2;e++){
          const int key=key0+8*n+2*t+e;
          if(live&&key<keys&&key<=seq+tok)v=fmaxf(v,sx[n][2*rh+e]*sc);
        }
        mx[rh]=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1));
        mx[rh]=fmaxf(mx[rh],__shfl_xor_sync(0xffffffffu,mx[rh],2));
      }
    };
    for(int j=0;j<nr;j++){
      const int rr=r+j;float sx[16][4];
      mb_wait(&kfull[rr%ST],(rr/ST)&1);protect(sx);wg_fence();
      qk128(sx,Q,Ks+(rr%ST)*16384+hh*8192);wg_commit();wg_wait<0>();protect(sx);
      reduce(sx,lo+128*(ka+j),seq);
      __syncwarp();if(lane==0)mb_arrive(&empty[rr%ST]);
    }
    r+=nr;
    if(virt){float sx[8][4];protect(sx);wg_fence();qk(sx,Q,KN);wg_commit();wg_wait<0>();protect(sx);reduce(sx,seq,seq+p.L);}
    const int slot=S.partB[s]+(cta-cta_of(S.segP[s],G,T));
    for(int rh=0;rh<2;rh++)if(t==0){
      const int row=w*16+g+8*rh;
      p.part[((int64_t)slot*128+hh*64+row)*PR+64]=mx[rh];
    }
  }
'''+c[end:]
    a=c.index('__global__ void __launch_bounds__(256, 4) vq8_combine(');b=c.index('\ntemplate <int ST> constexpr int smem_bytes()',a)
    c=c[:a]+'''__global__ void __launch_bounds__(256,4) vq8_combine(VP p){
      asm volatile("griddepcontrol.launch_dependents;");
      asm volatile("griddepcontrol.wait;":::"memory");
      const int seg=blockIdx.y,lane=threadIdx.x&31;
      const int rr=blockIdx.x*8+(threadIdx.x>>5);
      if(rr>=2*p.L*8)return;
      const int r=(rr>>3)/p.L*64+(rr>>3)%p.L*8+(rr&7);
      const int pb=sched_area(p)[seg],np=sched_area(p)[seg+1]-pb;
      float mx=-INFINITY;
      for(int j=lane;j<np;j+=32)mx=fmaxf(mx,p.part[((int64_t)(pb+j)*128+r)*PR+64]);
      for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
      const int b=seg/p.HP,hp=seg%p.HP,hh=r>>6,row=r&63,tok=row>>3,head=(2*hp+hh)*8+(row&7);
      if(lane==0)((float*)p.out)[(b*p.L+tok)*p.HQ+head]=mx==-INFINITY?0.f:mx+0.00390625f;
    }
'''+c[b:]
    c=one(c,'cfg.gridDim = dim3(2 * p.L * p.wpr, p.NS);','cfg.gridDim = dim3(2 * p.L, p.NS);')
    return c

def compact(c,stages):
    # New-token shared storage is needed only after history has drained. Reuse
    # the dead history stage then, freeing4KiB so ST3 can fit4CTAs/SM.
    c=one(c,'uint8_t* sNK = smem + 2 * STAGES * TILE_BYTES;','uint8_t* sNK = (L==1)?sK:smem + 2 * STAGES * TILE_BYTES;')
    c=one(c,'uint8_t* sNV = sNK + 16 * ROWB;','uint8_t* sNV = (L==1)?sV:sNK + 16 * ROWB;')
    c=one(c,'reinterpret_cast<__half*>(sNV + 16 * ROWB)','reinterpret_cast<__half*>((L==1)?smem+2*STAGES*TILE_BYTES:sNV+16*ROWB)')
    c=one(c,'  if (last) {  // zero the new-token rows past L','  if (last && L!=1) {  // zero the new-token rows past L')
    c=one(c,'      *reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;','      if constexpr(L!=1)*reinterpret_cast<__nv_fp8x2_storage_t*>(dst) = f8;')
    needle='''  if (last && (L > 1 || wh == 0)) {
    compute_tile<L, 2>'''
    c=one(c,needle,'''  if constexpr(L==1){if(last){
    __threadfence_block();__syncthreads();
    const int r=tid/8,col=tid%8;
    const long long row=p.cache_loc[b];
    const int off=r*ROWB+((col^swz(r))<<4);
    cp_async16(smem_u32(sNK+off),p.k_cache+row*p.row_bytes+kvg*ROWB+col*16,r==0?16:0);
    cp_async16(smem_u32(sNV+off),p.v_cache+row*p.row_bytes+kvg*ROWB+col*16,r==0?16:0);
    cp_commit();cp_wait<0>();__syncthreads();
  }}
  if (last && (L > 1 || wh == 0)) {
    compute_tile<L, 2>''')
    c=one(c,'2 * STAGES * TILE_BYTES + 2 * 16 * ROWB','2 * STAGES * TILE_BYTES + (L==1?0:2*16*ROWB)')
    for text in ('launch_t<1, 4>','occupancy_t<1, 4>','max_cluster_t<1, 4>'):
        assert text in c;c=c.replace(text,text.replace('4>',str(stages)+'>'))
    return c

def main():
    import pvsum_build
    v=(o/'VQ8_CUDA.cu').read_text();v=pvsum_build.cuda(v)
    (o/'A_vq.cu').write_text(v);(o/'F_vq.cu').write_text(fixed(v));(o/'B_vq.cu').write_text(bound(v));(o/'last.cu').write_text(last())
    q=json.loads((o/'base_bundle.json').read_text())['q1']
    for stages in (3,4):(o/('C'+str(stages)+'.cu')).write_text(compact(q,stages))
    meta={p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in (o/'A_vq.cu',o/'F_vq.cu',o/'B_vq.cu',o/'last.cu')}
    (o/'kernel_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
if __name__=='__main__':main()
