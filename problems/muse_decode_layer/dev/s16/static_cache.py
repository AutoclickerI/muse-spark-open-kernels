"""Cache only successful exact-source/flags Hopper compiles; all Python checks still run."""
import ast
import hashlib
import inspect
import json
import tempfile
import fcntl
from pathlib import Path

H=Path(__file__).resolve().parent


def install(local_eval):
    original=local_eval.hopper_compile
    code=inspect.getsource(original).replace('cf.ThreadPoolExecutor(8)','cf.ThreadPoolExecutor(2)')
    namespace=dict(local_eval.__dict__);exec(compile(code,'<s16-bounded-native-compile>','exec'),namespace)
    compiler=namespace['hopper_compile']
    cache=H/'cache';cache.mkdir(exist_ok=True)
    implementation=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    def check(path):
        source=path.read_text();tree=ast.parse(source);templated=set();flags={};nodes={}
        for n in ast.walk(tree):
            if isinstance(n,ast.Attribute) and n.attr in ('format','replace','format_map') and isinstance(n.value,ast.Name):templated.add(n.value.id)
            elif isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mod) and isinstance(n.left,ast.Name):templated.add(n.left.id)
            if isinstance(n,ast.Call) and getattr(n.func,'attr',getattr(n.func,'id',''))=='load_inline':
                kw={k.arg:k.value for k in n.keywords};v=kw.get('cuda_sources')
                names=[e.id for e in (v.elts if isinstance(v,(ast.List,ast.Tuple)) else [v]) if isinstance(e,ast.Name)]
                extra=[]
                for name,prefix in (('extra_cuda_cflags',''),('extra_include_paths','-I')):
                    try:extra += [prefix+str(x) for x in ast.literal_eval(kw[name])]
                    except (KeyError,ValueError):pass
                for name in names:flags[name]=[x for x in extra if not x.startswith(('-gencode','-arch','--generate-code'))]
        for n in tree.body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
                name=n.targets[0].id;text=n.value.value
                if '#include' in text and (name in flags or '__global__' in text or 'torch/extension.h' in text) and name not in templated:nodes[name]=n
        rows={};keys={}
        for name,n in nodes.items():
            key=hashlib.sha256(json.dumps([implementation,n.value.value,flags.get(name,[])],sort_keys=True).encode()).hexdigest();keys[name]=key
            p=cache/(key+'.json')
            if p.exists():
                row=json.loads(p.read_text());assert row['ok'];rows[name]=dict(row,name=name,msg='exact source/flags compile cache')
        if len(rows)<len(nodes):
            lines=source.splitlines(True)
            for name in sorted(rows,key=lambda name:nodes[name].lineno,reverse=True):
                n=nodes[name];lines[n.lineno-1:n.end_lineno]=[name+' = ""\n']
            with tempfile.TemporaryDirectory(prefix='native-',dir=cache) as tmp:
                p=Path(tmp)/path.name;p.write_text(''.join(lines))
                fresh=compiler(p)
            for row in fresh:
                rows[row['name']]=row
                if row['ok']:(cache/(keys[row['name']]+'.json')).write_text(json.dumps(row)+'\n')
        assert set(rows)==set(nodes)
        print('S16_STATIC',len(rows),'exact CUDA units',flush=True)
        return [rows[name] for name in nodes]
    def locked(path):
        with (cache/'compile.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return check(path)
    local_eval.hopper_compile=locked
