"""Instrument the losing 294fe739 recipe; do not change its projection bodies."""
import ast
import base64
import hashlib
import lzma
from pathlib import Path

HERE=Path(__file__).resolve().parent


def assignment(src,name):
    return ast.literal_eval(next(n.value for n in ast.parse(src).body if isinstance(n,ast.Assign)
        and getattr(n.targets[0],'id','')==name))


payload=(HERE/'lab_producer_hybrid.py').read_text()
assert hashlib.sha256(payload.encode()).hexdigest().startswith('294fe739')
source=assignment(payload,'PG_SOURCE')
cpp=assignment(source,'FIXED_O_CUDA')
traced=cpp.replace('pid*3','pid*8')
traced=traced.replace('stamps[pid*8+1]=stamps[pid*8+2]=0;',
    'for(int p=1;p<8;++p)stamps[pid*8+p]=0;')
traced=traced.replace('  __threadfence();__syncthreads();',
    '  if constexpr(PROFILE)if(tid==0)stamps[pid*8+3]=gt();\n  __threadfence();__syncthreads();')
traced=traced.replace('  if(tid==0)arrive64(count);',
    '  if(tid==0){arrive64(count);if constexpr(PROFILE)stamps[pid*8+4]=gt();}')
traced=traced.replace('    int group=pid-(C-G);',
    '    if constexpr(PROFILE)if(tid==0)stamps[pid*8+5]=gt();\n    int group=pid-(C-G);')
assert cpp!=traced and source.count(repr(cpp))==1
source=source.replace(repr(cpp),repr(traced),1)
source=source.replace('STAMP+pid*3','STAMP+pid*8').replace('(self.ctas,3)','(self.ctas,8)').replace('(512,3)','(512,8)')
old='        tl.store(STAMP+pid*8,_pg_clock())'
source=source.replace(old,old+'\n        for phase in tl.static_range(3,8):tl.store(STAMP+pid*8+phase,0)')
old='POST=POST)\n            tl.debug_barrier()'
assert source.count(old)==1
source=source.replace(old,'POST=POST)\n            if TRACE:tl.store(STAMP+pid*8+3,_pg_clock())\n            tl.debug_barrier()')
old="            tl.atomic_add(COUNT,1,sem='acq_rel',scope='gpu')"
assert source.count(old)==1
source=source.replace(old,old+'\n            if TRACE:tl.store(STAMP+pid*8+4,_pg_clock())')
old='            if ROLE==1 and group==0:'
assert source.count(old)==1
source=source.replace(old,'            if TRACE:tl.store(STAMP+pid*8+5,_pg_clock())\n'+old)
old='                _ps_topk(P,IDS,WTS,LIDS,BIAS,START,token,T=T,S=S,DEP=dep)'
assert source.count(old)==1
source=source.replace(old,old+'\n                if TRACE:tl.store(STAMP+pid*8+6,_pg_clock())')
old='                    if old_route==generation*T-1:_pg_meta(LIDS,META,START,T=T)'
assert source.count(old)==1
source=source.replace(old,'                    if old_route==generation*T-1:\n                        _pg_meta(LIDS,META,START,T=T)\n                        if TRACE:tl.store(STAMP+pid*8+7,_pg_clock())')

body=payload[:payload.index('PG_CASES=')].replace(repr(assignment(payload,'PG_SOURCE')),repr(source),1)
helpers=(HERE/'lab_body.py').read_text().split('def lab(deadline,weights,state=None):')[0]
helpers=helpers.replace("            assert not set(mapping)&ids,('nested capture',name)",
    '            ids-=set(mapping)  # inner boundary/route nodes retain their attribution')
helpers=helpers.replace("lambda a:'o_b1' if a[4]=='b1' else 'unpool_b2'",
    "lambda a:('o_b1' if a[4]=='b1' else 'unpool_b2') if a[2]['pg_ops'][(a[0].shape[0],a[4])].fuse else ('o' if a[4]=='b1' else 'unpooler')")
helpers=helpers.replace("lambda a:'pool_route',lambda a:'route' in roles",
    "lambda a:'pool_route' if a[1]['pg_ops'][(a[2]['config']['tokens'],'route')].fuse else 'pool_router',lambda a:'route' in roles")
body+=helpers+'\n'+(HERE/'mechanism_body.py.inc').read_text()
compile(body,'mechanism','exec')
(HERE/'lab_mechanism_body.py').write_text(body)
tag=hashlib.sha256(body.encode()).hexdigest()
bootstrap=('import base64,lzma,linecache,hashlib\n_PG_MECH_BODY=lzma.decompress(base64.b85decode('
    +repr(base64.b85encode(lzma.compress(body.encode(),preset=9)).decode())+')).decode()\n'
    +'assert hashlib.sha256(_PG_MECH_BODY.encode()).hexdigest()=='+repr(tag)+'\n'
    +'_PG_MECH_FILE="<pg-mechanism-'+tag[:12]+'>"\n'
    +'linecache.cache[_PG_MECH_FILE]=(len(_PG_MECH_BODY),None,_PG_MECH_BODY.splitlines(True),_PG_MECH_FILE)\n'
    +'exec(compile(_PG_MECH_BODY,_PG_MECH_FILE,"exec"),globals())\n'
    +'_PG_MECH_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _PG_MECH_RUN(deadline,weights,state)\n')
path=HERE/'lab_mechanism.py';path.write_text(bootstrap)
print(path,hashlib.sha256(bootstrap.encode()).hexdigest(),'body',tag)
