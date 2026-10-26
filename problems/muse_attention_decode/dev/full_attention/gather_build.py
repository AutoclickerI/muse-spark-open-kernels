"""Full q1 page-row transport: contiguous quads, aligned vector page-ID loads."""
import ast
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;D=H/'decode';out=H/'gather';out.mkdir(exist_ok=True)
s=(D/'ref.py').read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA')
c=ast.literal_eval(node.value);a=c.index('  auto load_rows =');b=c.index('  auto issue =',a)
c=c[:a]+r'''  auto load_rows = [&](int tile, int (&rows)[4]) {
    const int tb=c_lo+tile*T+4*lt;
    if(p.ps==1 && tile<ntiles && tb+3<c_hi && tb+7<p.max_pages){
      const uintptr_t addr=reinterpret_cast<uintptr_t>(pt+tb);
      const int shift=(addr>>2)&3;
      const uintptr_t aligned=addr&~uintptr_t(15);
      uint4 x,y;
      asm volatile("ld.global.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(aligned):"memory");
      if(shift){
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(y.x),"=r"(y.y),"=r"(y.z),"=r"(y.w):"l"(aligned+16):"memory");
      }
      if(shift==0){rows[0]=x.x;rows[1]=x.y;rows[2]=x.z;rows[3]=x.w;}
      else if(shift==1){rows[0]=x.y;rows[1]=x.z;rows[2]=x.w;rows[3]=y.x;}
      else if(shift==2){rows[0]=x.z;rows[1]=x.w;rows[2]=y.x;rows[3]=y.y;}
      else{rows[0]=x.w;rows[1]=y.x;rows[2]=y.y;rows[3]=y.z;}
    }else{
#pragma unroll
      for(int k=0;k<4;k++){
        const int t=tb+k;
        rows[k]=(tile<ntiles && t<c_hi)?(pt[t>>p.ps_log2]*p.ps+(t&(p.ps-1))):-1;
      }
    }
  };
'''+c[b:]
old='      const int t = lt + 16 * k;';assert c.count(old)==1;c=c.replace(old,'      const int t = 4 * lt + k;')
lines=s.splitlines(True);lines[node.lineno-1:node.end_lineno]=['CUDA = '+repr(c)+'\n'];driver=''.join(lines)
(out/'driver.py').write_text(driver);(out/'gather.cu').write_text(c)
(out/'manifest.json').write_text(json.dumps(dict(driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
    cuda_sha256=hashlib.sha256(c.encode()).hexdigest(),source='frozen decode ref driver; only load_rows/issue mapping changes',
    target='B16/32 q1 full, no sliding dispatch changes',promotion=False),indent=2)+'\n')
print(hashlib.sha256(c.encode()).hexdigest())
