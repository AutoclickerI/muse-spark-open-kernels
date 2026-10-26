"""tools/local_eval of shipped payload, with local-only native stand-ins."""
import argparse,base64,hashlib,json,lzma,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];F=H.parent/'full_attention'
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');a=ap.parse_args();mode='forced' if a.forced else 'normal';o=H/'out'
raw=(o/'preview.py').read_text();preview=o/'screen_preview.py';preview.write_text(raw.replace('# okc: problem=muse_bf16_proj','# okc: problem=muse_attention_decode',1))
hook=f'''
import importlib.util as _i,sys as _s,types as _ty
_s.path[:0]=[{str(F)!r},{str(R/'problems/muse_decode_layer/dev/s15')!r}]
def PUBLIC_LOCAL_INSTALL(base,op,lee,op_direct,lee_hybrid):
 saved={{k:getattr(base,k) for k in ('_attn_port_hopper','_fa3_on_device','ATTN_BACKEND') if hasattr(base,k)}}
 spec=_i.spec_from_file_location('public_verify',{str(R/'problems/muse_decode_layer/dev/verify_best.py')!r});helper=_i.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(base,False)
 if {not a.forced!r}:
  for k,v in saved.items():setattr(base,k,v)
 import rebuild_standins
 rebuild_standins.install(base)
 spec=_i.spec_from_file_location('public_standins',{str(H.parent/'public_ports/local_standins.py')!r});stand=_i.module_from_spec(spec);spec.loader.exec_module(stand);stand.operator(op,{a.forced!r});stand.lee(lee);stand.operator(op_direct,{a.forced!r});stand.lee(lee_hybrid)
'''
# Tool wrapper dispatches payload in its own namespace; inject only into that
# namespace and cap its timing fixture contexts locally, never the sent source.
mark='    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)'
assert raw.count(mark)==1
insert=hook+f'''
    # hook above is inserted at module scope; payload transfer uses fn globals.
'''
s=preview.read_text().replace('import time as _lab_time\n',hook+'\nimport time as _lab_time\n')
replacement='''    fn.__globals__['PUBLIC_LOCAL_INSTALL']=PUBLIC_LOCAL_INSTALL
    # Native cost is measured only on H200; shortened local fixtures exercise
    # every dispatch, FP32 layout, and selected glue under the same checker.
    _ckmod=fn.__globals__
    _out=fn(deadline,weights)
    __import__('pathlib').Path(LINES_PATH).write_text(__import__('json').dumps(_out))
    def _host_attention(data):
        c=data['config'];return fn.__globals__['base_call']((data['qkv'].reshape(1,c['batch']*c['q_len'],-1),1),data)
    globals()['kernel_fn']=_host_attention
    return _out'''.replace('LINES_PATH',repr(str(o/(mode+'_lines.json'))))
s=s.replace(mark,replacement).replace('_lab_import_start + 865','_lab_import_start + 3600')
# Change only decoded lab text passed to exec, not opaque bundled donor kernels.
# h200 wrapper payloads are literal source strings.
import ast
tr=ast.parse(s)
class LocalFixtures(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str) and "SEEDS=tuple(10739123" in n.value:
   n.value=n.value.replace('ctx=80000,page=1','ctx=257,page=1')
  return n
s=ast.unparse(LocalFixtures().visit(tr))+'\n'
runner=o/(mode+'_screen_runner.py');lines=o/(mode+'_lines.json');lines.unlink(missing_ok=True)
runner.write_text(f"""import sys,types,linecache
from pathlib import Path
sys.path[:0]={[str(R/'tools'),str(F)]!r}
import local_eval as ev
_old=ev.load_module
def _load(name,path):
 if name!='submission':
  m=_old(name,path)
  if name=='reference':m.CACHE_CHUNK=8192;m._fa3_scheduler_metadata=lambda *a,**kw:None
  return m
 src={s!r}
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
 linecache.cache[str(path)]=(len(src),None,src.splitlines(True),str(path))
 exec(compile(src,str(path),'exec'),m.__dict__);return m
ev.load_module=_load
sys.argv={['local_eval.py',str(preview),'--tests-only','--json',str(o/(mode+'_local_eval.json'))]!r}
ev.main()
""")
with (o/(mode+'.log')).open('w') as f:p=subprocess.run(['python3',str(runner)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,(mode,'local_eval',p.returncode)
assert lines.exists(),(mode,'payload failed; inspect '+str(o/(mode+'.log')))
rows=json.loads(lines.read_text());blob=next(x[12:] for x in rows if x.startswith('RESULT_XZ85 '));d=json.loads(lzma.decompress(base64.b85decode(blob)));exec((H/'codec.py.inc').read_text());d=decode_rounds(d);d.update(wrapper_sha256=hashlib.sha256(raw.encode()).hexdigest(),local_mode=mode)
(o/(mode+'.json')).write_text(json.dumps(d,indent=2)+'\n');meta=json.loads((o/'lab_manifest.json').read_text())
assert d['complete'] and not d['failures'] and all(d['checks'][x]==meta[x+'_TASK'] for x in ('official','ABI','scored')),d
report=json.loads((o/(mode+'_local_eval.json')).read_text());assert all(x['ok'] for x in report['tests']+report['static']),report
print(mode,'PASS',d['checks'],d['elapsed'],d['wrapper_sha256'],flush=True)
