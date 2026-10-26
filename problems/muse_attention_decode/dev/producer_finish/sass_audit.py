"""S19 producer geometry and native SASS; no inference of dynamic ALU idle."""
import collections,concurrent.futures,hashlib,json,linecache,re,subprocess,sys,sysconfig
from pathlib import Path
import torch
from torch.utils.cpp_extension import include_paths
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'setup_tree'));import pack_setup
s=pack_setup.base();fn='<pf-S19-source>';linecache.cache[fn]=(len(s),None,s.splitlines(True),fn);m={};exec(compile(s,fn,'exec'),m);ns=m['_nf_vendor']()
inst=(1,1,1,2,2,0,256,0,1280,8192,1,4,8,4)
sources={'d1_d9':ns['v18d_v18d_native_cuda_source']([inst])}
sub,insts,plans=m['sgq_source'](False,m['SGQ_SELECTED']);sources['v1_SGQ']=sub['v18s_cuda_source'](insts)
q32=ns['tt_v10_m32x__Q_CUDA'];q32=re.sub(r'case 1:return m32x_q_launch_t<[^\n]+\n','',q32);sources['d32_q32']=q32
sources['finish']=(H/'finish.cu').read_text()
def compile_one(item):
    name,src=item;p=H/'out'/(name+'.cu');p.write_text(src)
    flags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-DTORCH_EXTENSION_NAME=pf_native','-DTORCH_API_INCLUDE_EXTENSION_H','-isystem',sysconfig.get_paths()['include']]
    flags += [a for path in include_paths(device_type='cuda') for a in ('-isystem',path)]
    run=subprocess.run(['nvcc',*flags,'-cubin',str(p),'-o',str(p.with_suffix('.cubin'))],capture_output=True,text=True);log=run.stdout+run.stderr;p.with_suffix('.log').write_text(log);assert run.returncode==0,(name,log[-3000:])
    sass=subprocess.check_output(['nvdisasm','--print-line-info',str(p.with_suffix('.cubin'))],text=True);p.with_suffix('.sass').write_text(sass)
    resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.cubin'))],text=True)
    ops=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_.]+)',sass))
    # A static dependency audit: the actual producers already wait for
    # cluster partials before STG output. No Q-head normalization/RoPE is
    # fused; no profiler stall counter establishes idle ALU.
    examples=[]
    for prefix in ('LDG.','HMMA.','QGMMA.','STAS.','SYNCS.','STG.','MUFU.','F2FP.'):
        found=[l.strip() for l in sass.splitlines() if re.search(r'\b'+re.escape(prefix),l)]
        examples.extend(found[:3])
    out=dict(name=name,sha256=hashlib.sha256(src.encode()).hexdigest(),resources=resources,opcodes=dict(ops),examples=examples,static_only=True)
    print('PASS',name,resources.replace('\n',' ')[-250:],flush=True);return out
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(compile_one,sources.items()))
    out=dict(base_sha256=hashlib.sha256(s.encode()).hexdigest(),rows=rows,columns_per_coded_tile=16,columns_per_Q_head=64,
        conclusion='The selected coded producers finish 16 columns per cluster, not a full64-channel head. A head finalizer needs4 sibling tiles, in addition to K reduction. A global last-CTA finalizer starts after the stream; idle ALU during earlier weight loads does not hide that dependent tail. Per-head pipelining requires another coordination/scheduling design and is unmeasured.',
        source_evidence={'d9':'row0=... WR*RT*16; RED2 waits remote K partials before emit_compact; nonowner rank returns.','q32':'RED5 partitions token tiles across cluster ranks and emits16-column rows after cluster partial reduction.','v1':'WR1/RT1 exact12 S4 producer; four sibling16-column tiles form one64-channel head.'})
    (H/'sass_audit.json').write_text(json.dumps(out,indent=2)+'\n')
