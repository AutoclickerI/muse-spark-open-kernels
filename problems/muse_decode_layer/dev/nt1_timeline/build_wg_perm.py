"""Match the activation layout to NT1's register-A K permutation: fuse stores into router/W13, vectorize copies."""
import ast,hashlib
from pathlib import Path
from build_wg import cuda_source,cell_source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def source():
 old=cuda_source();cuda=old
 cuda=cuda.replace('const __nv_bfloat16* xr[NT/4];bool xv[NT/4];','constexpr int NJ=(NT+15)/16;const __nv_bfloat16* xr[NJ];bool xv[NJ];')
 cuda=cuda.replace('for(int j=0;j<NT/4;++j){int r=t/32+j*4;', 'for(int j=0;j<NJ;++j){int r=t/8+j*16;')
 a=cuda.index('for(int j=0;j<NT/4;++j){int r=t/32+j*4,k=');b=cuda.index('\n    asm volatile("cp.async.commit_group;',a)
 cuda=cuda[:a]+'''for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;
     if(r<NT)wg_cp(bs+slot*32*64+r*64+(k^((r&7)*8)),xr[j]+ki*64+k,16,xv[j]);}'''+cuda[b:]
 cuda=cuda.replace('int ch=mt*32+w*8+(l>>2);float scale=', 'int ch=mt*32+w*8+(l>>2);ch=(ch&~63)|(((ch>>2)&3)*16+((ch>>4)&3)*2+((ch>>1)&1)*8+(ch&1));float scale=')
 body=cell_source().replace(repr(old),repr(cuda),1).replace('wg_','wgp_').replace('WG_','WGP_')
 body=body.replace('64 output rows, adaptive 8/16/32 expert-token columns.', '64 output rows, adaptive 8/16/32 expert-token columns; expects router-permuted activations.')
 return body,cuda.replace('wg_','wgp_').replace('WG_','WGP_')
def build():
 body,cuda=source()
 for p,s in ((DEV/'wgp_cell.py',body),(DEV/'wgp.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
