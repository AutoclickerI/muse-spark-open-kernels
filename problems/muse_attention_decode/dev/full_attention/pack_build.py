"""Consume scores into packed P immediately, instead of a live FP32 P matrix."""
import ast,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'pack';o.mkdir(exist_ok=True)
base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
node=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA')
c=ast.literal_eval(node.value)
a=c.index('#pragma unroll\n  for (int rh = 0; rh < 2; rh++) {\n    const float mn = fmaxf(m[rh], mx[rh] * sc);')
b=c.index('\nDI void rescale(',a)
c=c[:a]+'''  float bias[2],sum0[2]={},sum1[2]={};
#pragma unroll
  for(int rh=0;rh<2;rh++){
    const float mn=fmaxf(m[rh],mx[rh]*sc);
    const float safe=mn==-INFINITY?0.f:mn;
    alpha[rh]=ex2(m[rh]-safe);m[rh]=mn;bias[rh]=8.f-safe;
  }
#pragma unroll
  for (int n = 0; n < NB; n+=2) {
#pragma unroll
    for(int rh=0;rh<2;rh++){
      const float p0=ex2(fmaf(s[n][2*rh],sc,bias[rh]));
      const float p1=ex2(fmaf(s[n][2*rh+1],sc,bias[rh]));
      const float p2=ex2(fmaf(s[n+1][2*rh],sc,bias[rh]));
      const float p3=ex2(fmaf(s[n+1][2*rh+1],sc,bias[rh]));
      // Native PV's RS register n+rh owns these two adjacent n-blocks.
      // Each score dies after conversion; no FP32 probability matrix survives.
      pk[n+rh]=pack4(p0,p1,p2,p3);
      sum0[rh]+=p0;sum1[rh]+=p1;sum0[rh]+=p2;sum1[rh]+=p3;
    }
  }
#pragma unroll
  for(int rh=0;rh<2;rh++)l[rh]=l[rh]*alpha[rh]+(sum0[rh]+sum1[rh]);
}
'''+c[b:]
lines=base.splitlines(True);lines[node.lineno-1:node.end_lineno]=['VQ8_CUDA = '+repr(c)+'\n'];s=''.join(lines)
tag=hashlib.sha256(c.encode()).hexdigest()[:12];s=s.replace('muse_vq8_9a23ca9a98dd_','muse_vq8_'+tag+'_')
(o/'I.py').write_text(s);(o/'I.cu').write_text(c)
p=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'))
(o/'I.patch').write_text(p)
meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest(),cuda_sha256=hashlib.sha256(c.encode()).hexdigest(),patch_sha256=hashlib.sha256(p.encode()).hexdigest())
(o/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
