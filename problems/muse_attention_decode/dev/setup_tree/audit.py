"""Reproduce S19 source identity and line-attributed SASS evidence from native.py."""
import ast,collections,hashlib,json,re
from pathlib import Path
import kernels_new as k
from pack_setup import base
H=Path(__file__).resolve().parent
def main():
    values={}
    for n in ast.parse(base()).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if getattr(t,'id','') in ('r2_KERNELS','wr_KERNELS'):values[t.id]=ast.literal_eval(n.value)
    d=values['r2_KERNELS']['dsm'];v=values['wr_KERNELS']['STREAM2']
    # Only the old multi-shape host dispatch is pruned. Device bodies are
    # byte-identical; original verify RAW2 is the entire exact S19 source.
    a=d.index('#define RES(');b=d.index('#undef RES',a)+len('#undef RES');d=d[:a]+'return resource<1,1,128>(cs);'+d[b:]
    a=d.index('#define GO(');b=d.index('#undef GO',a)+len('#undef GO');d=d[:a]+'run<1,1,128>(p,(cudaStream_t)stream);'+d[b:]
    assert d==k.source('d') and v==k.source('v')
    for n in ('dA','vA','dH','vH'):
        s=(H/'out'/f'{n}.cu').read_text();sass=(H/'out'/f'{n}.sass').read_text();ranges={}
        anchors=[('query_decode','DI void prep_decode(','DI void prep_verify('),('query_verify','DI void prep_verify(','DI void prep_new(' if n[0]=='v' else 'DI void prob('),('input_pairs','DI void input_pairs(','__device__ __noinline__'),('copy_issue','  auto issue=','  mark(p,0)' if n[0]=='v' else '  issue(0,0)'),('page_prefetch','  int preg[','  auto issue='),('cp_async','DI void cp16(','DI void commit('),('fp8_pair','DI uint16_t fp8pair(','DI uint32_t pack4(')]
        for key,start,end in anchors:
            try:lo=s[:s.index(start)].count('\n')+1;hi=s[:s.index(end,s.index(start)+len(start))].count('\n')+1
            except ValueError:continue
            ops=collections.Counter();examples=[];ln=0;source_file=''
            for line in sass.splitlines():
                m=re.search(r'//## File "([^"]+)", line (\d+)',line)
                if m:source_file=m.group(1);ln=int(m.group(2))
                m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_.]+)',line)
                if m and Path(source_file).name==n+'.cu' and lo<=ln<hi:
                    ops[m.group(1)]+=1
                    if len(examples)<12 and re.search(r'LDG|SHFL|MUFU|F2FP|SYNCS',m.group(1)):examples.append(dict(line=ln,sass=line.strip()))
            ranges[key]=dict(source_lines=[lo,hi-1],opcodes=dict(ops),examples=examples)
        (H/f'{n}_sass_audit.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(s.encode()).hexdigest(),S19_device_body_identity=n.endswith('A'),ranges=ranges,note='Static instruction sites, not dynamic cycles. Inline helper locations may be outside their call-site ranges; helpers are listed separately.'),indent=2)+'\n')
    print('S19 device identities and line-attributed SASS audited')
if __name__=='__main__':main()
