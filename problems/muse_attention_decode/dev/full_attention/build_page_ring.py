"""Next transport form: coordinate cp.async ring off the QK/PV issue path."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'page_ring';o.mkdir(exist_ok=True)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();node=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='ATTN_PORT_CUDA_SRC');original=ast.literal_eval(node.value)
meta={}
for PD in (4,8):
 c=original
 def one(a,b):
  global c
  assert c.count(a)==1,(a[:60],c.count(a));c=c.replace(a,b)
 # This is a proposed source until its TASK screens and native residency pass.
 one('  auto load_rows = [&](int tile, int (&rows)[4]) {',f'''  constexpr int PD={PD};
  __shared__ __align__(128) int page_ring[PD][64];
  const bool use_ring=(L==1 && !p.window);
  auto page_issue=[&](int tile){{
    if(threadIdx.x<64){{
      const int key=c_lo+tile*T+threadIdx.x;
      const int clamped=min(key,max(seq-1,0));
      const int*src=pt+(clamped>>p.ps_log2);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(&page_ring[tile%PD][threadIdx.x])),"l"(src):"memory");
    }}
  }};
  if(use_ring){{
    for(int j=0;j<PD;j++)page_issue(j);
    cp_commit();cp_wait<0>();__syncthreads();
  }}
  auto load_rows = [&](int tile, int (&rows)[4]) {{
    if(use_ring){{
      for(int k=0;k<4;k++)rows[k]=tile; // slot index, no global page dependency
      return;
    }}''')
 one('      const long long r = rows[k] < 0 ? 0 : rows[k];', '''      const int key=c_lo+rows[k]*T+t;
      const bool valid=use_ring?(key<c_hi):(rows[k]>=0);
      const int page=use_ring?page_ring[rows[k]%PD][t]:0;
      const long long r=use_ring?(page*p.ps+(key&(p.ps-1))):(rows[k]<0?0:rows[k]);''')
 c=c.replace('rows[k] < 0 ? 0 : 16','valid ? 16 : 0')
 # A page slot is refilled only after every warp read its current coordinates.
 needle='''      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, valid ? 16 : 0);
    }
  };'''
 one(needle,'''      cp_async16(smem_u32(sV + off), vbase + r * p.row_bytes, valid ? 16 : 0);
    }
    if(use_ring){__syncthreads();page_issue(rows[0]+PD);}
  };''')
 lines=base.splitlines(True);lines[node.lineno-1:node.end_lineno]=['ATTN_PORT_CUDA_SRC = '+repr(c)+'\n'];s=''.join(lines)
 name='R'+str(PD);(o/(name+'.py')).write_text(s);(o/(name+'.cu')).write_text(c)
 patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'));(o/(name+'.patch')).write_text(patch)
 meta[name]=dict(source_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(c.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),page_lookahead=PD,added_shared=PD*256)
(o/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
