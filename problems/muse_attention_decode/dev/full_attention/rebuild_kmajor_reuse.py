"""Receipt response: retain native QK; reclaim its dead K tile for scores."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];out=H/'rebuild'
base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
s=(out/'kmajor.py').read_text();n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='ATTN_PORT_CUDA_SRC');c=ast.literal_eval(n.value)
old='''    __shared__ __align__(128) float score[2][64][8];
#pragma unroll
    for(int hh=0;hh<2;hh++){
      float x[4];key_query(x,Ks,Q8+hh*512,hh);
      const int row=16*(threadIdx.x>>5)+(lane>>2),col=2*(lane&3);
      score[hh][row][col]=x[0];score[hh][row][col+1]=x[1];
      score[hh][row+8][col]=x[2];score[hh][row+8][col+1]=x[3];
    }
    __syncthreads();'''
new='''    float x[2][4];
#pragma unroll
    for(int hh=0;hh<2;hh++)key_query(x[hh],Ks,Q8+hh*512,hh);
    // Both QK products have consumed K. No prefetch overwrites this ring slot
    // until compute_tile returns. The next stage has a distinct slot.
    __syncthreads();
    auto score=reinterpret_cast<float (*)[64][8]>(const_cast<uint8_t*>(Ks));
#pragma unroll
    for(int hh=0;hh<2;hh++){
      const int row=16*(threadIdx.x>>5)+(lane>>2),col=2*(lane&3);
      score[hh][row][col]=x[hh][0];score[hh][row][col+1]=x[hh][1];
      score[hh][row+8][col]=x[hh][2];score[hh][row+8][col+1]=x[hh][3];
    }
    __syncthreads();'''
assert c.count(old)==1;c=c.replace(old,new)
lines=s.splitlines(True);lines[n.lineno-1:n.end_lineno]=['ATTN_PORT_CUDA_SRC = '+repr(c)+'\n'];s=''.join(lines)
tag=hashlib.sha256(c.encode()).hexdigest()[:12]
s=s.replace('muse_decode_attn_port_kmajor_9d9302df9638_', 'muse_decode_attn_port_kmajor_reuse_'+tag+'_')
compile(s,'kmajor_reuse','exec');(out/'kmajor_reuse.py').write_text(s);(out/'kmajor_reuse.cu').write_text(c)
patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'));(out/'kmajor_reuse.patch').write_text(patch)
meta=dict(source_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(c.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),response_to='1bab0b2d0dc14931',change='Reuse dead cached K ring tile for score remap; -4096 static shared bytes, four extra live score values and one CTA barrier.',gather_unchanged=True,split_plan_unchanged=True,bytes=dict(dynamic_q1=73984,old_extra_static=6144,new_extra_static=2048),native_validation='See native_kmajor_reuse.json for compiled register/spill/instruction proof; TASK admission is separate.')
(out/'kmajor_reuse_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
