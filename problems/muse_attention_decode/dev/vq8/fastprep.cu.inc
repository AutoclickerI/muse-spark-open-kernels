// Cooperative row preparation: four lanes hold one row, eight adjacent BF16 pairs per lane.
// A warp prepares eight rows concurrently; each 128-thread head group covers all 64 Q rows
// in two batches. This replaces sixteen serial full-warp reductions in the original prologue.
DI void prologue(const VP& p, uint8_t* Q, uint8_t* KN, uint8_t* VN, int b, int hp, int hh, int seq, int w, int lane,
                 bool virt, float ksc, float vsc) {
  const int tid = threadIdx.x & 127, sub = tid & 3, row = tid >> 2;
  const int kvh = 2 * hp + hh;
  int tk[16], cl[16]; bool vd[16];
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int r = row + 32 * j, tok = r >> 3;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int a = 8 * j + i;
      vd[a] = tok < p.L;
      tk[a] = b * p.L + (vd[a] ? tok : 0);
      cl[a] = (kvh * 8 + (r & 7)) * 64 + 16 * sub + 2 * i;
    }
  }
  float2 x[16];
  input_pairs<16>(p, tk, cl, vd, x);
#if VQ8_CLK
  if (threadIdx.x == 0 && !clk_area()[44]) clk_area()[44] = gtime();
#endif
  // RoPE is shared across eight GQA heads. Compute each (token, frequency) once;
  // the new-K buffer is scratch until Q preparation finishes.
  float2* csbuf = reinterpret_cast<float2*>(KN);
  if (p.window) {
    for (int i=tid; i<p.L*32; i+=128) {
      float sn, cs;
      __sincosf((float)(seq+i/32)*p.inv_freq[i&31], &sn, &cs);
      csbuf[i]=make_float2(cs,sn);
    }
  }
  wg_bar(1+hh);
#pragma unroll
  for (int j = 0; j < 2; j++) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += x[8*j+i].x*x[8*j+i].x + x[8*j+i].y*x[8*j+i].y;
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float norm = rsqrtf(ss * (1.f/64.f) + 1e-5f);
    const int r = row + 32*j, tok = r >> 3;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float cs = 1.f, sn = 0.f;
      if (p.window) { const float2 v=csbuf[min(tok,p.L-1)*32+8*sub+i]; cs=v.x; sn=v.y; }
      const float2 v = x[8*j+i];
      const float a = v.x*norm, z = v.y*norm;
      *(uint16_t*)(Q + off(r, 16*sub+2*i)) = fp8pair(a*cs-z*sn, a*sn+z*cs);
    }
  }
  wg_bar(1+hh);  // no query reader uses csbuf when the new K overwrites it
  int kt[4], kc[4]; bool valid[4];
#pragma unroll
  for (int i=0; i<2; i++) {
    const int tok=2*w+i;
    valid[i]=valid[2+i]=virt && tok<p.L;
    kt[i]=kt[2+i]=b*p.L+(tok<p.L?tok:0);
    kc[i]=(p.HQ+kvh)*64+2*lane;
    kc[2+i]=(p.HQ+p.HK+kvh)*64+2*lane;
  }
  float2 kv[4];
  input_pairs<4>(p,kt,kc,valid,kv);
  const int64_t loc[2]={p.loc[kt[0]],p.loc[kt[1]]};
#pragma unroll
  for (int i=0; i<2; i++) {
    if (!valid[i]) break;
    const int tok=2*w+i;
    float2 k=kv[i], v=kv[2+i];
    const float norm=rsqrtf(wsum(k.x*k.x+k.y*k.y)*(1.f/64.f)+1e-5f);
    float cs=1.f, sn=0.f;
    if (p.window) __sincosf((float)(seq+tok)*p.inv_freq[lane],&sn,&cs);
    const float a=k.x*norm,z=k.y*norm;
    const uint16_t kq=fp8pair((a*cs-z*sn)/ksc,(a*sn+z*cs)/ksc), vq=fp8pair(v.x/vsc,v.y/vsc);
    *(uint16_t*)(p.kc+(loc[i]*p.HK+kvh)*64+2*lane)=kq;
    *(uint16_t*)(p.vc+(loc[i]*p.HK+kvh)*64+2*lane)=vq;
    *(uint16_t*)(KN+off(tok,2*lane))=kq;
    const int pos=4*(tok>>1)+(tok&1);
    VN[off(2*lane,pos)]=(uint8_t)vq;
    VN[off(2*lane+1,pos)]=(uint8_t)(vq>>8);
  }
}
