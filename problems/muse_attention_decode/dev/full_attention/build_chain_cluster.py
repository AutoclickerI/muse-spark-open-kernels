"""One native fixed-split control and one distributed-DSM fused epilogue."""
import difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;o=H/'chain';b=json.loads((H/'phases/base_bundle.json').read_text())
def one(c,a,z):
 assert c.count(a)==1,(a[:70],c.count(a));return c.replace(a,z)
def source(fused):
 c=b['VQ8_CUDA'];mark='template <int ST>\n__global__ void __launch_bounds__(384, 1) vq8_main(VP p) {'
 c=one(c,mark,(H/'chain_cluster.cu.inc').read_text()+'\n'+mark)
 a=c.index('  if (warp == 0) {',c.index('__launch_bounds__(384, 1) vq8_main'));z=c.index('  if(tid<32)',a)
 c=c[:a]+'''  if(tid==0){
    const int s=blockIdx.y,seq=p.seq[s/p.HP],cs=p.min_tiles;
    const int nc=(seq+127)>>7,T=nc+1,ka=blockIdx.x*T/cs,kb=(blockIdx.x+1)*T/cs;
    S.segSeq[s]=seq;S.segP[s]=0;S.segP[s+1]=T;S.partB[s]=s*cs;
    S.G=cs;S.T=T;S.npo=1;S.nreal=max(0,min(kb,nc)-ka);S.po_s[0]=s;S.po_ka[0]=ka;S.po_kb[0]=kb;
    if(blockIdx.x==0){sched_area(p)[s]=s*cs;if(s==p.NS-1)sched_area(p)[s+1]=p.NS*cs;}
  }
'''+c[z:]
 c=one(c,'    const int slot = S.partB[s] + (cta - cta_of(S.segP[s], G, T));','    const int slot = s * p.min_tiles + blockIdx.x;')
 if fused:
  c=one(c,'    const int slot = s * p.min_tiles + blockIdx.x;','''    // Both query WGs have drained before Q/K stage storage is reused.
    asm volatile("bar.sync 5,256;":::"memory");
    const int slot = s * p.min_tiles + blockIdx.x;''')
  c=one(c,'float* dst = p.part + ((int64_t)slot * 128 + hh * 64 + w * 16 + g + 8 * rh) * PR;', 'extern __shared__ __align__(1024) uint8_t scratch[];\n      float* dst=(float*)(scratch+CLKPAD)+(hh*64+w*16+g+8*rh)*PR;')
  a=c.index('  if (warp >= 8) {',c.index('__launch_bounds__(384, 1) vq8_main'));z=c.index('\n}\n\n// wpr',a)
  c=c[:a]+'''  if(warp>=8)producer<ST>(p,S,Ks,VR,VT,kfull,full,empty,pages);
  else consumer<ST>(p,S,Qs,Kn,VTn,Ks,VT,kfull,full,empty);
  __syncthreads();cluster_reduce(p,sm);
'''+c[z:]
 c=one(c,'cfg.gridDim = dim3(p.grid);','cfg.gridDim = dim3(p.min_tiles,p.NS);')
 c=one(c,'cudaLaunchAttribute attr[1];','cudaLaunchAttribute attr[2];\n  attr[1].id=cudaLaunchAttributeClusterDimension;attr[1].val.clusterDim={static_cast<unsigned>(p.min_tiles),1,1};')
 c=one(c,'cfg.numAttrs = pdl ? 1 : 0;', 'cfg.numAttrs=2;')
 # Even the non-PDL case uses a legal cluster launch; its ordinary edge is still
 # respected by griddepcontrol.wait (and the stream when PDL is not eligible).
 c=c.replace('  constexpr int smem = smem_bytes<ST>();\n  cudaFuncSetAttribute(vq8_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);','  constexpr int smem = smem_bytes<ST>();\n  cudaFuncSetAttribute(vq8_main<ST>,cudaFuncAttributeNonPortableClusterSizeAllowed,1);\n  cudaFuncSetAttribute(vq8_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);')
 if fused:
  a=c.index('  cudaLaunchConfig_t cfg = {};',c.index('cudaLaunchKernelEx(&cfg, vq8_main<ST>, p);'));z=c.index('\n}',a);c=c[:a]+c[z:]
 # Control combine covers the exact fixed per-segment slot map.
 return c
c=b['VQ8_CUDA'];a=c.index('  cudaLaunchConfig_t cfg = {};',c.index('cudaLaunchKernelEx(&cfg, vq8_main<ST>, p);'));z=c.index('\n}',a);c=c[:a]+c[z:]
(o/'N.cu').write_text(c)
meta={}
for name,f in [('G',False),('F',True)]:
 c=source(f);(o/(name+'.cu')).write_text(c);patch=''.join(difflib.unified_diff(b['VQ8_CUDA'].splitlines(True),c.splitlines(True),fromfile='a/S19_VQ8_CUDA',tofile='b/'+name))
 (o/(name+'.cuda.patch')).write_text(patch);meta[name]=dict(sha256=hashlib.sha256(c.encode()).hexdigest(),fused=f)
(o/'cluster_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
