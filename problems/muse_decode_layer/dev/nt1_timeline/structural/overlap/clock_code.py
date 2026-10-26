"""Lab-only trace:per-warp phase residence and first16 issue/math timestamps;not issue-utilization counters."""
def instrument(cu,ws=False):
 def sub(a,b,n=1):
  nonlocal cu
  assert cu.count(a)==n,(a,cu.count(a));cu=cu.replace(a,b)
 #256slots:original64;16issue/16mathbegin/16mathend;12warps x10 phase/count fields.
 cu=cu.replace('(size_t)c * 64','(size_t)c * 256').replace('(size_t)c*64','(size_t)c*256')
 anchor='  auto dump = [&](bool finalizer) {'
 if ws:
  sub('  __shared__ uint64_t probe[TRACE ? 64 : 1];','  uint64_t* probe=TRACE?trace+(size_t)c*256:nullptr;')
  init='  uint64_t(*wclock)[10]=TRACE?reinterpret_cast<uint64_t(*)[10]>(trace+(size_t)c*256+128):nullptr;\n'
 else:init='  __shared__ uint64_t wclock[TRACE?12:1][TRACE?10:1];\n'
 init+='''  if constexpr(TRACE){if(lane==0)for(int q=0;q<10;++q)wclock[warp][q]=0;
    if(tid==0)for(int q=64;q<128;++q)trace[(size_t)c*256+q]=0;
    __syncthreads();}
  auto ck_start=[&](){uint64_t t=0;if constexpr(TRACE)if(lane==0)t=tclk();return t;};
  auto ck_add=[&](int q,uint64_t t){if constexpr(TRACE)if(lane==0)wclock[warp][q]+=tclk()-t;};
  int clock_tile=0;
'''
 sub(anchor,init+anchor)
 #Publish counters once,apart from producer warp's separate early exit.
 anchor='      if (tid == 0) { probe[8] = tg();'
 sub(anchor,'      if(lane==0)for(int q=0;q<10;++q)trace[(size_t)c*256+128+warp*10+q]=wclock[warp][q];\n'+anchor)
 a='''      if (pass == 1) {
        uint64_t wait0 = 0;''';sub(a,a.replace('        uint64_t','        const uint64_t ck_r=ck_start();\n        uint64_t'))
 anchor='if constexpr (TRACE) { if (tid == 0) probe[16] += tg() - wait0; }';sub(anchor,anchor+'\n        ck_add(1,ck_r);')
 anchor='''#pragma unroll
      for (int j = 0; j < NT; ++j) {''';sub(anchor,'      const uint64_t ck_b=ck_start();\n      if constexpr(TRACE)if(lane==0)wclock[warp][8]++;\n'+anchor)
 #The segment fill ends immediately before first-W2-clock andraw wait.
 anchor='''    }
    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp) probe[4] = tg(); }''';sub(anchor,'      ck_add(0,ck_b);\n'+anchor)
 sub('    mbar_wait(&fullb[st], ph);','    const uint64_t ck_w=ck_start();\n    mbar_wait(&fullb[st], ph);\n    ck_add(2,ck_w);')
 sub('    const uint8_t* tile = stages + st * TILE_BYTES;','''    const uint64_t ck_m=ck_start();
    if constexpr(TRACE)if(tid==0&&clock_tile<16)trace[(size_t)c*256+80+clock_tile]=tg();
    const uint8_t* tile = stages + st * TILE_BYTES;''')
 sub('    __syncwarp();\n    if (lane == 0) mbar_arrive(&emptyb[st]);','''    ck_add(3,ck_m);
    if constexpr(TRACE){if(tid==0&&clock_tile<16)trace[(size_t)c*256+96+clock_tile]=tg();if(lane==0)wclock[warp][7]++;}
    __syncwarp();
    if (lane == 0) mbar_arrive(&emptyb[st]);''')
 sub('    float* rp = red +','    const uint64_t ck_s=ck_start();\n    float* rp = red +')
 sub('    bar_consumers();\n    if constexpr(TRACE)if(tid==0)probe[52+pass]', '    bar_consumers();\n    ck_add(4,ck_s);\n    if constexpr(TRACE)if(tid==0)probe[52+pass]')
 anchor='    const float* rb = red +';sub(anchor,'    const uint64_t ck_e=ck_start();\n'+anchor)
 sub('    if constexpr(TRACE)if(tid==0){probe[56+pass]', '    ck_add(6,ck_e);\n    if constexpr(TRACE)clock_tile++;\n    if constexpr(TRACE)if(tid==0){probe[56+pass]')
 #Call-site timestamps include the helper's issue+bookkeeping;do not call them transaction completion times.
 call='prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);'
 if not ws:
  cu=cu.replace('else '+call,'else {const int ci=TRACE?rt->pre.index:0;const uint64_t ct=ck_start();'+call+'ck_add(5,ct);if constexpr(TRACE)if(ci<16&&ci<total_count)trace[(size_t)c*256+64+ci]=tg();}')
  #Initial fill call has no else prefix.
  cu=cu.replace('        '+call,'        {const int ci=TRACE?rt->pre.index:0;const uint64_t ct=ck_start();'+call+'ck_add(5,ct);if constexpr(TRACE)if(ci<16&&ci<total_count)trace[(size_t)c*256+64+ci]=tg();}')
 else:
  a='for(int i=0;i<total;++i){prefetch_one(rt,total,first,aa2,nstage,stages,fullb,emptyb,w13p,w2p);'
  sub(a,'for(int i=0;i<total;++i){const uint64_t ct=ck_start();prefetch_one(rt,total,first,aa2,nstage,stages,fullb,emptyb,w13p,w2p);ck_add(5,ct);if constexpr(TRACE)if(i<16)trace[(size_t)c*256+64+i]=tg();')
  sub('    return;\n  }\n  asm volatile("setmaxnreg.inc','    if constexpr(TRACE)if(lane==0)for(int q=0;q<10;++q)trace[(size_t)c*256+128+warp*10+q]=wclock[warp][q];\n    return;\n  }\n  asm volatile("setmaxnreg.inc')
 #Measure copy address/issue separately from waiting for an empty slot.
 a=cu.index('__device__ __forceinline__ void prefetch_one(');b=cu.index('// ---- flag-templated',a);helper=cu[a:b]
 helper=helper.replace('const uint8_t* w13p, const uint8_t* w2p) {','const uint8_t* w13p, const uint8_t* w2p,uint64_t* pc=nullptr) {')
 helper=helper.replace('  int st=p.stage,', '  uint64_t cp0=0,cw0=0,cwait=0;if(pc)asm volatile("mov.u64 %0, %%clock64;":"=l"(cp0)::"memory");\n  int st=p.stage,')
 helper=helper.replace('  mbar_wait(&emptyb[st], ph^1u);','  if(pc)asm volatile("mov.u64 %0, %%clock64;":"=l"(cw0)::"memory");\n  mbar_wait(&emptyb[st], ph^1u);\n  if(pc){uint64_t now;asm volatile("mov.u64 %0, %%clock64;":"=l"(now)::"memory");cwait=now-cw0;pc[2]+=cwait;}')
 helper=helper.replace('  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;','  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n  if(pc){uint64_t now;asm volatile("mov.u64 %0, %%clock64;":"=l"(now)::"memory");pc[5]+=now-cp0-cwait;pc[9]++;}')
 cu=cu[:a]+helper+cu[b:]
 import re
 cu=re.sub(r'prefetch_one\(([^;\n]+)\);',lambda m:'prefetch_one('+m[1]+',TRACE?&wclock[warp][0]:nullptr);',cu)
 cu=cu.replace('const uint64_t ct=ck_start();','').replace('ck_add(5,ct);','')
 return cu
