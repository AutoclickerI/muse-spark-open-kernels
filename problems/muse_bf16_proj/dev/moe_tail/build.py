"""S15 route/finish rebuild: original expert binaries remain shared controls."""
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
S15='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
ARMS=('A','A2','R','F4','F8','C','E','P','PF')
def fun(s,name):return ast.get_source_segment(s,[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name][-1])
def once(s,a,b):assert s.count(a)==1,(a[:60],s.count(a));return s.replace(a,b,1)
def main():
 base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();assert hashlib.sha256(base.encode()).hexdigest()==S15
 cu=ast.literal_eval(next(n.value for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CUDA'))
 a=cu.index('template<bool META>__global__');b=cu.index('// SPLIT blocks',a);route=cu[a:b].replace('template<bool META>','').replace('route_live(','route(').replace('if constexpr(META)','if constexpr(true)')
 route=once(route,'   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}', '''   unsigned peers=__match_any_sync(~0u,e);int lead=__ffs(peers)-1,first=0;
   if((t&31)==lead && e>=0&&e<32)first=atomicAdd(count+e,__popc(peers));
   first=__shfl_sync(~0u,first,lead);
   if(e>=0&&e<32){rank=first+__popc(peers&((1u<<(t&31))-1));SLOTS[e*M+rank]=i;}''')
 a=cu.index('template<int SPLIT>__global__');b=cu.index('}  // namespace mwbr',a);finish=cu[a:b].replace('template<int SPLIT>','template<int SPLIT,bool PDL>')
 a=finish.index(' int e=');b=finish.index(' for(int d=16;',a)
 finish=finish[:a]+''' int e=load_i(IDS+tok*8+w)-start,p=-1;float weight=load_f(TW+tok*8+w);
 if(e>=0&&e<32)p=load_i(OFF+e)+load_i(POS+tok*8+w);
 wait<PDL>();float sum=0;
 if(p>=0)sum=dependent<PDL>(SS+(size_t)p*64+l)+dependent<PDL>(SS+(size_t)p*64+l+32);
'''+finish[b:];finish=once(finish,'*TW[tok*8+w]', '*weight')
 finish=once(finish,'*(const float4*)(Y+(size_t)pp[j]*4096+c)','dependent4<PDL>(Y+(size_t)pp[j]*4096+c)')
 native=(H/'tail.cu.inc').read_text().replace('MT_ROUTE',route).replace('MT_FINISH',finish)
 pmz=cu[cu.index('namespace pmz {'):]
 anchor=' constexpr int NB=8*NT,RED=NT*4;'
 pmz=once(pmz,anchor,'''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
 if constexpr(!FIRST)asm volatile("griddepcontrol.launch_dependents;":::"memory");
#endif
'''+anchor)
 pmz='\n'.join(native.splitlines()[:6])+'''\n#define BR_CUDA_CHECK MT_CK
#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("mt pmz args");}while(0)
'''+pmz
 plan=fun(base,'nf_plan').replace('def nf_plan(data,base,native,streams):','def mt_plan(data,base,native,streams,arm):')
 plan=once(plan,'    light_grid=4','    light_grid=4;tail=mt_module();release=arm in (\'E\',\'P\',\'PF\');pm=mt_pmz_module() if release else None')
 plan=once(plan,'_=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)',"_=owners\n        if arm in ('R','C'):tail.mt_route(rp,m,start,torch.cuda.current_stream().cuda_stream)\n        else:mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)")
 plan=once(plan,'mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)','(pm if release else mod).pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)')
 plan=once(plan,"        _=owners;mod.br_finish(fp,m,start,4,torch.cuda.current_stream().cuda_stream);return out",'''        _=owners
        if arm in ('R','E'):mod.br_finish(fp,m,start,4,torch.cuda.current_stream().cuda_stream)
        else:tail.mt_finish(fp,m,start,{'F4':4,'F8':8,'C':4,'P':10,'PF':14}[arm],torch.cuda.current_stream().cuda_stream)
        return out''')
 code='MT_CUDA='+repr(native)+'\nMT_PMZ_CUDA='+repr(pmz)+'\nMT_ARMS='+repr(ARMS)+'\nMT_ARM="A"\n'+(H/'runtime.py.inc').read_text()+'\n'+plan+'\n'
 code+='''
_mt_original_plan=nf_plan
def nf_plan(data,base,native,streams):
    if data['hidden_states'].shape[0] not in (128,256) or MT_ARM in ('A','A2'):return _mt_original_plan(data,base,native,streams)
    return mt_plan(data,base,native,streams,MT_ARM)
_mt_original_prepare=prepare
def prepare(weights):
    state=_mt_original_prepare(weights);mt_module();mt_pmz_module();return state
'''
 compile(code,'moe_tail','exec');(H/'runtime_generated.py.inc').write_text(code);(H/'native_generated.py.inc').write_text(native);(H/'pmz_generated.py.inc').write_text(pmz)
 sources={}
 for arm in ARMS[2:]:
  s=base+'\n'+code+'\nMT_ARM='+repr(arm)+'\n';(H/(arm+'_trial.py')).write_text(s);sources[arm]=hashlib.sha256(s.encode()).hexdigest()
 (H/'manifest.json').write_text(json.dumps(dict(s15_sha256=S15,native_sha256=hashlib.sha256(native.encode()).hexdigest(),pmz_sha256=hashlib.sha256(pmz.encode()).hexdigest(),runtime_sha256=hashlib.sha256(code.encode()).hexdigest(),variants=sources,arms=ARMS),indent=2)+'\n')
 print('built',len(sources),'S15 arms; expert/weight/grids unchanged; native',hashlib.sha256(native.encode()).hexdigest()[:8])
if __name__=='__main__':main()
