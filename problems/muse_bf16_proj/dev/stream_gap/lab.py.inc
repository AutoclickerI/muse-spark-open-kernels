def sg_load():
    import linecache
    ns={'__name__':'sg_candidate'};file='<sg_candidate>';linecache.cache[file]=(len(SG_SOURCE),None,SG_SOURCE.splitlines(True),file)
    exec(compile(SG_SOURCE,file,'exec'),ns);return ns


def lab(deadline,weights,state=None):
    import time,math,statistics,hashlib
    begin=time.monotonic();m=sg_load();T=torch
    if weights is None or 'pooler' not in weights:
        gen=T.Generator(device='cuda').manual_seed(61791)
        def dense(n,k):return (T.randn((n,k),device='cuda',generator=gen)/math.sqrt(k)).bfloat16()
        weights=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),unpooler=dense(8192,4096),router=.045*T.randn((256,8192),device='cuda',generator=gen))
    st=m['prepare'](weights);arch=m['_nf_vendor']()['_nf_arch_flag']()
    probe=m['load_inline'](name='sg_stream_'+hashlib.sha256(SG_PROBE.encode()).hexdigest()[:12],cpp_sources='#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch_scan(int64_t,int64_t,int64_t,int64_t);\nint launch_nop(int,int64_t);\nint capture_count(int64_t);\n',cuda_sources=SG_PROBE,functions=['launch_scan','launch_nop','capture_count'],extra_cuda_cflags=['-O3',arch],no_implicit_headers=True,with_pytorch_error_handling=False)
    def graph(fn):
        fn();T.cuda.synchronize();g=T.cuda.CUDAGraph()
        with T.cuda.graph(g):
            out=fn();nodes=probe.capture_count(T.cuda.current_stream().cuda_stream)
        g.replay();T.cuda.synchronize();return g,out,nodes
    def times(gs,rounds=6):
        events=[]
        for r in range(rounds):
            for j in (range(len(gs)) if r%2==0 else range(len(gs)-1,-1,-1)):
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True);a.record();gs[j].replay();b.record();events.append((j,a,b))
        T.cuda.synchronize();out=[[] for _ in gs]
        for j,a,b in events:out[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(x) for x in out]
    def ci(rr):
        logs=[math.log(r) for r in rr];avg=statistics.mean(logs);e=2.364624*statistics.stdev(logs)/math.sqrt(len(logs))
        return math.exp(avg),math.exp(avg-e),math.exp(avg+e)
    nopgraphs=[graph(lambda n=n:probe.launch_nop(n,T.cuda.current_stream().cuda_stream)) for n in (0,1,2,4,8,16)]
    nop=times([x[0] for x in nopgraphs],12);empty=nop[0]
    lines=[f'META A=exact P12 2d5caebc; D=generic NF12 direct BF16 store; F=same generic FP32+cast; P=current producer; router=NF rt2 exact hi/lo',
           f'META8 heldout input seeds,6cold rounds;logical bytes are compulsory banks,not HBM counters;setup={time.monotonic()-begin:.1f}s']
    lines.append('NOP '+','.join(f'{n}:{v:.4f}' for n,v in zip((0,1,2,4,8,16),nop)))
    for nbytes in sorted({1<<20,4<<20,8<<20,12<<20,16<<20,24<<20,32<<20,48<<20,64<<20,96<<20,128<<20}):
        p=T.randint(-(1<<31),(1<<31)-1,(nbytes//4,),device='cuda',dtype=T.int32);out=T.empty((nbytes//16384,),device='cuda',dtype=T.int32)
        gg,_,_=graph(lambda:probe.launch_scan(p.data_ptr(),out.data_ptr(),p.numel(),T.cuda.current_stream().cuda_stream))
        gg.replay();T.cuda.synchronize();want=int(p.to(T.int64).sum().item())&0xffffffff;got=int(out.to(T.int64).sum().item())&0xffffffff;assert got==want,('stream checksum',nbytes,want,got)
        vals=[]
        for _ in range(8):vals.append(times([gg],8)[0]-empty)
        lines.append(f'STREAM {nbytes} {statistics.median(vals):.5f} [{min(vals):.5f},{max(vals):.5f}]')
    def producer(x,role):
        t=x.shape[0]
        if role=='router':return m['gemv'](x,st['router_pair'],out_dtype=T.float32)
        name='pool_router' if role=='pooler' else 'unpooler_slice' if role=='unpooler' and t<=8 else role
        return m['_project_variant'](x,st[name],name,st,m['PROJ_PINS'].get((t,name),'custom'),T.bfloat16)
    def weight_bytes(t,role,pair):
        name='pool_router' if role=='pooler' else 'unpooler_slice' if role=='unpooler' and t<=8 else role
        def size(v):return v.numel()*v.element_size()
        if role=='router':return size(st['router_pair'])
        if (t,name) in m['NF_PORT_SELECTED']:
            p=st['nf_port']['plans'].get((t,name),{})
            if 'wp' in p:return size(p['wp'])+size(p['esc'])
            if 'state' in p:return size(p['state']['weight'])
        if (t,name) in m['CD_ROWS']:
            p=st['coded_dense'][name];return sum(size(p[k]) for k in ('coded','scales','eoff','esc'))
        return size(st[name])
    count=0
    for t in (1,8,16,32,128,256):
        for role in ('qkv_gate','o','pooler','unpooler','router'):
            if time.monotonic()>deadline-20:return lines+[f'INCOMPLETE {count}/30']
            w=weights[role][:1024] if role=='unpooler' and t<=8 else weights[role];n,k=w.shape;x=T.randn((t,k),device='cuda',dtype=T.bfloat16);data=dict(x=x,config=dict(m=t,n=n,k=k,layer=role))
            labels=['A','AA','D']+(['F'] if role!='router' and t<=32 else [])+['P']
            saved=[graph(lambda:m['sg_run'](data,st,'A')),graph(lambda:m['sg_run'](data,st,'A')),graph(lambda:m['sg_run'](data,st,'direct'))]
            if 'F' in labels:saved.append(graph(lambda:m['sg_run'](data,st,'float')))
            saved.append(graph(lambda:producer(x,role)));vals=[[] for _ in saved];errors=[0.]*len(saved)
            pair=saved[-1][1];ab=weight_bytes(t,role,pair);db=ab
            if role!='router' and t<=32:
                p=st['sg_direct']['plans'][t,role];db=p['wp'].numel()*p['wp'].element_size()+p['esc'].numel()*p['esc'].element_size()
            elif role=='router':db=weights['router'].numel()*4
            for seed in (68137,29009,41879,73939,12517,94421,50207,86269):
                gen=T.Generator(device='cuda').manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=gen).bfloat16());x[-1]=0.;x[-1,(seed*131)%k]=16
                ref=x.double()@w.double().T
                for j,(g,out,_) in enumerate(saved):
                    g.replay();T.cuda.synchronize()
                    if labels[j]=='P':out=m['_p12_materialize'](out,role=='router')
                    diff=out.double()-ref;bound=1e-4 if role=='router' else .01
                    e=float(diff.norm()/ref.norm());avg=ref.norm()/math.sqrt(t);den=T.maximum(ref.norm(dim=1),avg*.1)
                    assert T.isfinite(out).all() and e<=bound and bool((diff.norm(dim=1)/den<=5*bound).all()),(t,role,labels[j],seed,e)
                    pe=float(diff[-1].norm()/ref[-1].norm());assert pe<=.001,(t,role,labels[j],'probe',pe);errors[j]=max(errors[j],e)
                got=times([a[0] for a in saved]);
                for j,v in enumerate(got):vals[j].append(v-empty)
            aa=ci([b/a for a,b in zip(vals[0],vals[1])]);dr=ci([b/a for a,b in zip(vals[0],vals[2])]);fr=ci([b/a for a,b in zip(vals[0],vals[3])]) if 'F' in labels else (0,0,0)
            med=[statistics.median(a) for a in vals]
            lines.append(f'R {t}:{role} A={med[0]:.4f} D={med[2]:.4f} F={(med[3] if "F" in labels else 0):.4f} P={med[-1]:.4f} dr={dr[0]:.6f},{dr[1]:.6f},{dr[2]:.6f} fr={fr[0]:.6f} aa={aa[0]:.6f},{aa[1]:.6f},{aa[2]:.6f} nodes={",".join(str(a[2]) for a in saved)} bytes={ab},{db} splits={pair[1]} e={max(errors):.1e}')
            count+=1
    return lines+[f'END complete={count}/30 elapsed={time.monotonic()-begin:.1f}s']
