"""Prepare a structural page-fetch candidate while the native phase probe runs."""
import ast
import hashlib
from pathlib import Path

H = Path(__file__).resolve().parent
tree = ast.parse((H / 'q1_runtime.py').read_text())
old = '''#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = c_lo + tile * T + lt + 16 * k;
      rows[k] = (tile < ntiles && t < c_hi) ? (pt[t >> p.ps_log2] * p.ps + (t & (p.ps - 1))) : -1;
    }'''
new = '''int pages[4], tokens[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int t = c_lo + tile * T + lt + 16 * k;
      const int safe_t = min(max(t, 0), max(seq - 1, 0));
      tokens[k] = t;
      // Volatile asm prevents sinking the read under the validity branch.
      asm volatile("ld.global.b32 %0,[%1];" : "=r"(pages[k]) : "l"(pt + (safe_t >> p.ps_log2)));
    }
#pragma unroll
    for (int k = 0; k < 4; ++k)
      rows[k] = (tile < ntiles && tokens[k] < c_hi) ? (pages[k] * p.ps + (tokens[k] & (p.ps - 1))) : -1;'''
for n in tree.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('Q1_BASE', 'Q1_CLOCK'):
        s = ast.literal_eval(n.value)
        assert s.count(old) == 1
        s = s.replace(old, new, 1)
        n.value = ast.Constant(s)
        if n.targets[0].id == 'Q1_BASE':(H / 'q1_pageflight.cu').write_text(s)
out = ast.unparse(tree) + '\n'
(H / 'q1_pageflight.py').write_text(out)
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
(H/'pageflight_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'q1_runtime.py').read_text(),'new':out})+'\n'+(H/'pageflight_lab.py.inc').read_text())
print('Page loads use safe unconditional addresses before validity selection;', hashlib.sha256(out.encode()).hexdigest())
