"""Controlled probes only. PRE arms move activation gathering outside timing and cannot promote."""
import ast,base64,hashlib,importlib.util,json,lzma,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parent
s=importlib.util.spec_from_file_location('act_builder',B/'actreuse/build.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
once=p.once

def cuda():
 base=p.source();prefix,alltiles=p.tiles(base)
 pack=alltiles[alltiles.index('template<int NT,bool F,int CS,int S>'):alltiles.index('template<int NT,bool F,int CS,int S>',alltiles.index('template<int NT,bool F,int CS,int S>')+10)]
 mcstart=alltiles.index('template<int NT,bool F,int CS,int S>',alltiles.index('template<int NT,bool F,int CS,int S>')+10);mcend=alltiles.index('template<int NT,bool F,int CS,int S>',mcstart+10);mc=alltiles[mcstart:mcend]
 early=pack.replace('template<int NT,bool F,int CS,int S>','template<int NT,bool F,int S,bool EARLY>').replace('void packed_tile(','void staged_tile(')
 early=early.replace('NJ=NT/16,BS=', 'NJ=NT/16,XS=S+1,BS=')
 early=early.replace(' int rank=blockIdx.x%CS;','')
 a=early.index(' auto issue=[&]');e=early.index('  if(t<64){',a)
 early=early[:a]+''' auto issue_x=[&](int k){int st=k%XS;
  if(wg==0)[&]<int...J>(std::integer_sequence<int,J...>){
   (copyx<NT,J,0>(bs,copy_root,st,count-row-t/8),...);
   (copyx<NT,J,1>(bs,copy_root,st,count-row-t/8),...);
  }(std::make_integer_sequence<int,NJ>{});copy_root+=128;
 };
 auto issue=[&](int k,int st){if(k+(EARLY?1:0)<32)issue_x(k+(EARLY?1:0));
'''+early[e:]
 early=once(early,' #pragma unroll\n for(int p=0;', ' if constexpr(EARLY)issue_x(0);asm volatile("cp.async.commit_group;":::"memory");\n #pragma unroll\n for(int p=0;')
 early=once(early,'int k=k0+st;if(k>=32)continue;','int k=k0+st;if(k>=32)continue;int sx=k%XS;')
 # Only descriptor reads use the X ring; Q/SF stay on their original three-stage ring.
 early=early.replace('bs+st*NT*KT','bs+sx*NT*KT').replace('desc0+(st*NT*KT','desc0+(sx*NT*KT')
 dup=mc.replace('template<int NT,bool F,int CS,int S>','template<int NT,bool F,int CS,int S,bool SYNC>').replace('void multicast_tile(','void duplicate_tile(')
 dup=dup.replace('chunk::cb();','if constexpr(SYNC)chunk::cb();else __syncthreads();')
 dup=once(dup,'if(threadIdx.x==0&&rank==0){','if(threadIdx.x==0){')
 dup=once(dup,'((1u<<CS)-1)','(1u<<rank)')
 dup=once(dup,'bytes.multicast::cluster','bytes')
 dup=once(dup,'[%4],%5;','[%4];')
 dup=once(dup,',"h"((unsigned short)(1u<<rank))','')
 cu=prefix+'\n#include <cudaTypedefs.h>\nnamespace probe {using namespace chunk;\n'+early+'\n'+dup+'\n'+(H/'kernels.cu.inc').read_text()+'\n'+(H/'host.cu.inc').read_text()
 return cu

def main():
 cu=cuda();(H/'probe.cu').write_text(cu)
 # Reuse only the qualified fixture/bank helpers and frozen S18/cell strings.
 old=(B/'actreuse/expanded.py').read_text();core=old[:old.index('ACT_NAMES=')]
 # Host-only metadata exposes the unchanged route for the explicit pre-gather oracle.
 cell=p.value(old,'ACT_CELL')
 cell=once(cell,'    def body():','    def route_only():\n        _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)\n    def gather_only():\n        _=owners;mod.ax_gather(gather_args,gx.data_ptr(),m,torch.cuda.current_stream().cuda_stream)\n    def body():')
 cell=once(cell,'return dict(steps=(route,body,finish),out=out,owners=owners)','return dict(steps=(route,body,finish),out=out,owners=owners,route_only=route_only,gather_only=gather_only)')
 core+='\nACT_CELL='+repr(cell)+'\n'+(B/'actreuse/runtime.py.inc').read_text().split('def act_counts')[0]
 cpp=p.value(p.source(),'P3B_CPP')+'void probe_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> probe_info(int64_t,int64_t);'
 body=core+'\nPROBE_CUDA='+repr(cu)+'\nPROBE_CPP='+repr(cpp)+'\n'+(H/'runtime.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text()
 compile(body,'<p3b-mechanism>','exec');blob=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode();digest=hashlib.sha256(body.encode()).hexdigest()
 packed='import base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=='+repr(digest)+'\n_file='+repr('<p3b-mechanism-'+digest[:12]+'>')+'\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,"exec"))\nLAB_RESOURCES={"compile_s":70,"run_s":90,"output_bytes":8200,"context_poison":"none"}\ndef lab(deadline,weights,state=None):return mechanism_lab(deadline,weights,state)\n'
 (H/'mechanism_payload.py').write_text(packed);(H/'expanded.py').write_text(body)
 (H/'manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(packed.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),bytes=len(packed),base_expanded_sha256='95df42fe7b3a31e66d60bc769bae15412c952ad77753b7d15b3beea3d89b3af4',prepacked_arms_never_promotable=True),indent=2)+'\n')
 print(len(packed),hashlib.sha256(cu.encode()).hexdigest())
if __name__=='__main__':main()
