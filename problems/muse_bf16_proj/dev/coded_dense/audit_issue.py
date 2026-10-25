"""Extract sm90 source with a line-info tag on the original steady-state loop; report SASS costs."""
import ast
import json
import re
import sys
from pathlib import Path

H=Path(__file__).resolve().parent

def extract():
 tree=ast.parse((H/'coded.py').read_text())
 src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CD_CUDA')
 src=src.replace('  for (int b = 0; b < nb; b += PIPE) {','  for (int b = 0; b < nb; b += PIPE) {\n#line 1000 "cd_hot"')
 mark='  }\n\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) {'
 assert src.count(mark)==1
 src=src.replace(mark,'  }\n#line 3000 "cd_epilogue"\n\n#pragma unroll\n  for (int rt = 0; rt < RT; ++rt) {')
 src=src.replace('CDG_INSTANCES',' '.join(f'CDG_CASE({nt},1,2,3)' for nt in (1,2,4)))
 (H/'issue_sm90.cu').write_text(src)

def summary():
    from pathlib import Path
    import re,json,collections,hashlib
    h=Path('problems/muse_bf16_proj/dev/coded_dense')
    out={'units':'SASS instructions executed by one warp per K64 block (two unrolled blocks in source); not dynamic counters','old':{},'wg':{}}
    pat=r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Za-z][A-Za-z0-9_.]+)\s'
    old=(h/'issue_sm90.sass').read_text();new=(h/'wg_sm90.sass').read_text()
    for nt in (1,2,4):
     sec=next(x for x in re.split(r'\.section\s+\.text\.',old) if x.startswith(f'_ZN3cdg10coded_gemvILi{nt}ELi1ELi2ELi3'))
     ops=collections.Counter(re.findall(pat,sec))
     out['old'][nt]=dict(HMMA_per_k64=ops['HMMA.16816.F32.BF16']/2,LDS128_per_k64=ops['LDS.128']/2,decoder_integer_per_weight=9/8,activation_address_ops_per_n8_k64=4,activation_regs_per_n8=8,accumulator_regs_per_n8=4)
    for sec in new.split('Function : ')[1:]:
     m=re.match(r'_ZN3cdg7wg_gemvILi(8|16|32)ELi(1|2)',sec)
     if not m:continue
     ops=collections.Counter(re.findall(pat,sec))
     out['wg'][f'T{m[1]}:WB{m[2]}']=dict(HGMMA_per_k64=sum(v for k,v in ops.items() if k.startswith('HGMMA'))/2,LDS128_per_k64=ops['LDS.128']/2,local_loads=sum(v for k,v in ops.items() if k.startswith('LDL')),local_stores=sum(v for k,v in ops.items() if k.startswith('STL')))
    assert len(out['wg'])==6
    assert all(r['LDS128_per_k64']==r['local_loads']==r['local_stores']==0 and r['HGMMA_per_k64']==4 for r in out['wg'].values()),out
    for fn,body in re.findall(r"Compiling entry function '([^']+)'.*?Function properties for [^\n]+\n(.*?)(?=ptxas info    : Compiling entry|\Z)",(h/'wg_sm90.log').read_text(),re.S):
     m=re.search(r'ILi(8|16|32)ELi(1|2)',fn);regs=int(re.search(r'Used (\d+) registers',body)[1]);stack=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body)
     assert stack and all(int(v)==0 for v in stack.groups()),fn
     out['wg'][f'T{m[1]}:WB{m[2]}'].update(registers=regs,stack_bytes=0,spills=0)
    out['wg_kernel_sha256']=hashlib.sha256((h/'wg_kernel.cu').read_bytes()).hexdigest()
    (h/'issue_model.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
 extract() if len(sys.argv)==1 else summary()
