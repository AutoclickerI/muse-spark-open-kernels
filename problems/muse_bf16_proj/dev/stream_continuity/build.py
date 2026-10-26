"""Source-pinned S17 continuity arms; all work remains inside the call."""
import ast,base64,difflib,hashlib,json,lzma,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
BASE='fea717603233f240b3e5d2c99b5ffdd0edb19d4264161489da44cc192b7f55dc'

def literal(s,name):return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)

def late_source(c,prefetch):
    helper='''
struct SCFuture {const char* p[2];uint64_t groups[2],stride[2],head[2];};
__device__ __forceinline__ void sc_future(SCFuture f){
  for(int d=0;d<2;++d)for(uint64_t g=blockIdx.x;g<f.groups[d];g+=gridDim.x)
    for(uint64_t off=0;off<f.head[d];off+=65536){
      unsigned n=(unsigned)min((uint64_t)65536,f.head[d]-off);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0],%1;"::"l"(f.p[d]+g*f.stride[d]+off),"r"(n):"memory");
    }
}
'''
    c=c.replace('namespace {','namespace {\n'+helper,1)
    assert c.count('const uint8_t* w13p, const uint8_t* w2p) {')==2
    c=c.replace('const uint8_t* w13p, const uint8_t* w2p) {','const uint8_t* w13p, const uint8_t* w2p, SCFuture future) {')
    a='  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);'
    z='  bulk_g2s(stages + st * TILE_BYTES, src, TILE_BYTES, &fullb[st], p.pol_first);'
    release='asm volatile("griddepcontrol.launch_dependents;":::"memory");'
    for mark,cond in ((a,'idx+1==total'),(z,'idx+1==total')):
        assert c.count(mark)==1
        c=c.replace(mark,mark+'\n  if('+cond+'){'+('sc_future(future);' if prefetch else '')+release+'}')
    # Include the explicit empty trace slot before the two immutable descriptors.
    mark='int M, int local_start, int nstage, uint64_t* __restrict__ trace) {'
    assert c.count(mark)==1;c=c.replace(mark,mark.replace('trace) {','trace, SCFuture future) {'))
    assert c.count('w13p, w2p);')==4;c=c.replace('w13p, w2p);','w13p, w2p, future);')
    mark='  cudaLaunchConfig_t cfg = {};\n  cfg.gridDim = dim3(grid);'
    init='''  SCFuture future{};
  if(v.size()==22)for(int d=0;d<2;++d){future.p[d]=(const char*)v[14+4*d];future.groups[d]=v[15+4*d];future.stride[d]=v[16+4*d];future.head[d]=v[17+4*d];}
'''
    assert c.count(mark)==1;c=c.replace(mark,init+mark)
    mark='(uint64_t*)(v.size() > 13 ? v[13] : 0)))';assert c.count(mark)==1;c=c.replace(mark,mark[:-2]+', future))')
    return c.replace('moe_var','moe_sc_late')

W_CODE='''
_sc_old_port=_port_cuda
_SC_PORT_AUDIT=[]
def _port_cuda(src,**kwargs):
    src=_sc_old_port(src,**kwargs)
    if not any(n in src for n in ('smallm2_gemv(', 'v18s_gemv(')):return src
    src=src.replace(': "l"(p));', ': "l"(p):"memory");')
    wait='  asm volatile("griddepcontrol.wait;" ::: "memory");\\n'
    assert src.count(wait)==1;src=src.replace(wait,'',1)
    start=src.index('    auto load = [&](int slot, int s) {')
    end=src.index('    if constexpr',src.index('    for (int s = 0; s < ST; ++s) load(s, s);',start))
    old=src[start:end];split=old.index('      if constexpr (!NOX')
    prefix=old[:split].replace('auto load =','auto loadw =',1)
    suffix=old[split:old.index('\\n    };',split)]
    new=prefix+'    };\\n    auto loadx = [&](int slot,int s) {\\n'+suffix+'\\n    };\\n'
    new+='    auto load = [&](int slot,int s){loadw(slot,s);loadx(slot,s);};\\n'
    order='smallm2_order_point' if 'smallm2_gemv(' in src else 'v18s_order_point'
    new+='#pragma unroll\\n    for(int s=0;s<ST;++s)loadw(s,s);\\n'+order+'(m,outp);\\n'+wait
    new+='#pragma unroll\\n    for(int s=0;s<ST;++s)loadx(s,s);\\n'
    src=src[:start]+new+src[end:]
    _SC_PORT_AUDIT.append(hashlib.sha256(src.encode()).hexdigest())
    return src
'''

def build():
    packed=(R/'problems/muse_decode_layer/cand_r3_best7_s17_final.py').read_text()
    node=next(n for n in ast.walk(ast.parse(packed)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    base=base64.b85decode(ast.literal_eval(node.args[0]));base=lzma.decompress(base).decode()
    assert hashlib.sha256(base.encode()).hexdigest()==BASE
    runtime='SC_CUDA='+repr((H/'prefetch.cu.inc').read_text())+'\n'+(H/'runtime.py.inc').read_text()
    sources={'A':base,'A2':base};meta={}
    for arm in ('E','P','W','F','B','L','X'):
        source=base
        if arm in ('E','P','X'):
            # Replace each selected native body in situ, keeping S17 NORM/PAIR
            # arithmetic and dispatch aliases rather than overriding them.
            for key in ('SMALLM_VAR_CUDA','_s16_epi_NORM_EPIN_CUDA','_s16_epi_PAIR_EPIT_CUDA'):
                old=literal(source,key);new=late_source(old,arm!='E')
                node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key)
                seg=ast.get_source_segment(source,node);source=source.replace(seg,key+' = '+repr(new),1)
            tree=ast.parse(source)
            for n in reversed(tree.body):
                if isinstance(n,ast.FunctionDef) and n.name in ('smallm_experts','_s16_epi_NORM_epin_experts','_s16_epi_PAIR_epit_experts'):
                    fn=ast.get_source_segment(source,n)
                    if 'ptrs = [v.data_ptr() for v in tensors]' not in fn:continue
                    mark="    mod.run(ptrs, m, data['config']['local_start'], stages, grid, torch.cuda.current_stream(dev).cuda_stream, flags)"
                    assert mark in fn
                    new=fn.replace(mark,"    if len(ptrs)==13:ptrs.append(0)\n    future=list(_SC_CURRENT_FUTURE);future += [0]*(8-len(future));ptrs += future\n"+mark)
                    source=source.replace(fn,new,1)
        suffix='\nSC_ARM='+repr(arm)+'\n'+runtime
        if arm in ('W','X'):suffix+='\n'+W_CODE
        source+=suffix;compile(source,'SC_'+arm,'exec');sources[arm]=source
        (H/(arm+'_trial.py')).write_text(source)
    for arm,source in sources.items():
        meta[arm]=dict(sha256=hashlib.sha256(source.encode()).hexdigest(),changed_vs_A=arm not in ('A','A2'))
    (H/'runtime_generated.py.inc').write_text(runtime)
    (H/'manifest.json').write_text(json.dumps(dict(base_sha256=BASE,shipped_base_sha256=hashlib.sha256(packed.encode()).hexdigest(),arms=meta,preselected='X',source_count=len(sources),scope='S17; unchanged arithmetic, actual consumer banks, selected NORM/PAIR preserved'),indent=2)+'\n')
    return sources
if __name__=='__main__':print({k:hashlib.sha256(v.encode()).hexdigest()[:8] for k,v in build().items()})
