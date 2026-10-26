"""K-major native FP8 QK in S15 q1's existing64-row gather/split ring."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];out=H/'rebuild';base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
n=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='ATTN_PORT_CUDA_SRC');c=ast.literal_eval(n.value)
def one(a,b):
 global c
 assert c.count(a)==1,(a[:120],c.count(a));c=c.replace(a,b)
one('template <int L, int NTC>\nDEVI void compute_tile', (H/'kmajor.cu.inc').read_text()+'\ntemplate <int L, int NTC, bool KM=false>\nDEVI void compute_tile')
one('need_mask, float scale_log2) {','need_mask, float scale_log2, const uint8_t* Q8=nullptr) {')
a=c.index('  // S = Q K^T.');z=c.index('\n  if (need_mask)',a);old=c[a:z]
new='''  if constexpr(KM && L==1 && NTC==4){
    __shared__ __align__(128) float score[2][64][8];
#pragma unroll
    for(int hh=0;hh<2;hh++){
      float x[4];key_query(x,Ks,Q8+hh*512,hh);
      const int row=16*(threadIdx.x>>5)+(lane>>2),col=2*(lane&3);
      score[hh][row][col]=x[0];score[hh][row][col+1]=x[1];
      score[hh][row+8][col]=x[2];score[hh][row+8][col+1]=x[3];
    }
    __syncthreads();
#pragma unroll
    for(int n=0;n<NTC;n++){
      s[0][n][0]=score[h][tb+8*n+2*q4][g];
      s[0][n][1]=score[h][tb+8*n+2*q4+1][g];
      s[0][n][2]=s[0][n][3]=0.f;
    }
    __syncthreads();
  }else{
'''+old+'\n  }\n'
c=c[:a]+new+c[z:]
# Specialize launch family instead of changing scored sliding/q8 kernels.
c=c.replace('template <int L, int STAGES>','template <int L, int STAGES, bool KM=false>')
c=c.replace('attn_main<L, STAGES>','attn_main<L, STAGES, KM>')
c=c.replace('compute_tile<L, NT>(st,','compute_tile<L, NT,KM>(st,')
one('                        need_mask, scale_log2);','                        need_mask, scale_log2,fixedQ8);')
one('  WarpState<L> st;', '''  uint8_t* fixedQ8=nullptr;
  if constexpr(KM){
    __shared__ __align__(1024) uint8_t q8[1024];fixedQ8=q8;
    // Convert this CTA's normalized query once, reused over every cache tile.
    for(int i=tid;i<1024;i+=128){
      int hh=i/512,rr=(i/64)&7,col=i&63;
      q8[hh*512+q8off(rr,col)]=__nv_cvt_float_to_fp8(__half2float(sQ[i]),__NV_SATFINITE,__NV_E4M3);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");__syncthreads();
  }
  WarpState<L> st;''')
# Preserve the original new-token path (only64 threads participate at q1).
one('(q_len == 1) ? launch_t<1, 4>(p, s) : launch_t<8, 3>(p, s);',
    '(q_len == 1) ? (p.window ? launch_t<1,4>(p,s) : launch_t<1,4,true>(p,s)) : launch_t<8,3>(p,s);')
# Original plan, gather and split count are supplied by unchanged S15 state.
lines=base.splitlines(True);lines[n.lineno-1:n.end_lineno]=['ATTN_PORT_CUDA_SRC = '+repr(c)+'\n'];s=''.join(lines)
s=s.replace("name=f'muse_decode_attn_port_{arch}'","name=f'muse_decode_attn_port_kmajor_"+hashlib.sha256(c.encode()).hexdigest()[:12]+"_{arch}'")
compile(s,'kmajor','exec');(out/'kmajor.cu').write_text(c);(out/'kmajor.py').write_text(s)
patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'));(out/'kmajor.patch').write_text(patch)
meta=dict(candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(c.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),gather_unchanged=True,split_plan_unchanged=True,target='q1 full cached64-key QK, N8queries/M64keys; original PV/newtoken path',promotion=False)
(out/'kmajor_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
