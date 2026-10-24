"""Standalone measurement of the direct, single-bank serving reader."""
import hashlib
from pathlib import Path
from build_moe_real_routing import namespace
from build_moe_v9_reuse import reference_generator
from build_shared_bank_bridge import DRIVER

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

READER_DRIVER=r'''
def lab(deadline, weights, host_state=None):
    lines=[]
    try:
        start=bridge_time.monotonic();device=weights['w13'].device
        stock=ma_prepare(weights);bank=stock['banks'][(0,32)]
        joined=[];shared_bank=dict(bank)
        for name in ('w13','w2'):
            q,s=bank[name],bank[name+'_s']
            data=torch.cat((q.view(torch.uint8).flatten(),s.view(torch.uint8).flatten()))
            count=q.nbytes
            shared_bank[name]=data[:count].view(torch.int32).view_as(q)
            shared_bank[name+'_s']=data[count:].view(torch.float8_e4m3fn).view_as(s)
            assert torch.equal(q,shared_bank[name])
            assert torch.equal(s.view(torch.uint8),shared_bank[name+'_s'].view(torch.uint8))
            joined.append(data)
        shared=dict(stock,banks={(0,32):shared_bank})
        gate=bank['w13_gs'].float()*(2.0**-119);down=bank['w2_gs'].float()*(2.0**-119)
        state=dict(w13p=joined[0],w2p=joined[1],mod=dr_compile_reader(v9_CUDA_SRC,v9_CPP_SRC),
                   gs13_gate=gate,gs13_up=gate,gs2=down,pnw=weights['post_norm_w'],
                   local_start=0,grid=torch.cuda.get_device_properties(device).multi_processor_count)
        # Native V9 is retained only as this lab's same-fixture control.
        t,j=torch.arange(512,device=device),torch.arange(8,device=device)
        rm1=torch.cat((t[:,None]*8+j,4096+t[:,None]*8+j),1)
        rm2=torch.arange(4096,device=device).view(256,16)
        native=dict(state,mod=v9__module())
        for name,rm in [('w13',rm1),('w2',rm2)]:
            out=torch.empty((32,rm.shape[0],36864),device=device,dtype=torch.uint8)
            for e in range(32):out[e]=v9__repack(weights[name][e],weights[name+'_sf'][e],rm)
            native[name+'p']=out
        routing=rg_generate_weights(1234,device=str(device),router_seed=5353)
        flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
        sizes=(8,16,24,32,40,48,56,64,80,96,112,128,144,160,176,192,208,224,240,256)
        for m in sizes+(512,753,16384):
            if bridge_time.monotonic()>deadline-20:
                return lines+[f'REJECT incomplete_before_M={m} budget_exhausted']
            ds=[]
            for seed in (9001,16920,24839):
                d=fx_make_fixture(routing,m,seed,mode='real',q_len=8 if m<=256 else 1)
                d['topk_ids']=torch.where(d['topk_ids']>=0,d['topk_ids'],255)
                ds.append({k:d[k] for k in ('hidden_states','topk_ids','topk_weights','config')})
            def direct(d):return v9__v9_call(d,state,return_dense=True)[0]
            def old(d):return v9__v9_call(d,native,return_dense=True)[0]
            funcs=[lambda d:ma_kernel_fn(d,stock)]
            funcs += [old,direct] if m<=256 else [lambda d:ma_kernel_fn(d,shared)]
            graphs=[bridge_graph(fn,ds[0]) for fn in funcs]
            maxdiff=rel=0.0
            for d in ds:
                outputs=[]
                valid=(d['topk_ids']<32).unsqueeze(-1)
                for graph,out,static in graphs:
                    for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(d[key])
                    graph.replay()
                    # Agreement uses stock's norm/combine, outside all timers.
                    y=torch.where(valid,out,0)
                    outputs.append(ma_fused_post_expert_norm_weighted_sum(
                        y,d['topk_weights'],norm_weight=stock['norm_w'],eps=1e-8,weight_first=True))
                torch.cuda.synchronize()
                if m<=256:
                    assert torch.equal(outputs[1],outputs[2]),(m,'native/direct V9 bit equality')
                for output in outputs[1:]:
                    diff=(output.float()-outputs[0].float()).abs()
                    maxdiff=max(maxdiff,float(diff.max()))
                    rel=max(rel,float(diff.norm()/outputs[0].float().norm().clamp_min(1e-20)))
                    assert rel<=.015,(m,rel,maxdiff)
            times=[[] for _ in graphs]
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for repeat in range(20):
                order=list(range(len(graphs)))
                order=order[repeat%len(order):]+order[:repeat%len(order)]
                for i in order:
                    for key in ('hidden_states','topk_ids','topk_weights'):
                        graphs[i][2][key].copy_(ds[repeat%len(ds)][key])
                    flush.sum();a.record();graphs[i][0].replay();b.record();b.synchronize()
                    times[i].append(a.elapsed_time(b)*1000)
            med=[bridge_statistics.median(x) for x in times]
            line=f'M={m} stock_us={med[0]:.3g} '
            if m<=256:line+=f'V9_native_us={med[1]:.3g} direct_us={med[2]:.3g} speedup={med[0]/med[2]:.3g} '
            else:line+=f'shared_Marlin_us={med[1]:.3g} speedup={med[0]/med[1]:.3g} '
            lines.append(line+f'rel={rel:.3g} maxdiff={maxdiff:.3g}')
            for graph,out,static in graphs:graph.reset()
        lines.append(f'DIRECT_READER_SUMMARY PASS shapes=23 fresh=3 cold_rotated20=1 extra_resident_bytes=0 prepare_s={bridge_time.monotonic()-start:.3g}')
    except Exception as exc:
        lines.append('REJECT '+type(exc).__name__+': '+str(exc)[:500])
    return lines
'''


def main():
    parts=['"""Direct V9 reader over stock Marlin NVFP4; no second serving bank."""\n']
    for prefix,path in (
        ('ma_','problems/muse_moe_w4a16/baseline.py'),
        ('v9_','serving/sglang/python/sglang/srt/layers/muse_spark_moe.py'),
        ('dr_','serving/sglang/python/sglang/kernels/ops/moe/muse_marlin_v9_reader.py'),
        ('fx_','problems/muse_decode_layer/moe_tune.py'),
    ):
        s=(ROOT/path).read_text()
        if prefix=='ma_':s=s[:s.index('    return fused_post_expert_norm_weighted_sum(')]+'    return y\n'
        if prefix=='fx_':s=s[:s.index('\ndef _graph_time')]
        parts.append(f'# Source {path} sha256={hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}\n'+namespace(s,prefix))
    parts += [reference_generator(),'\nimport torch\n',DRIVER[:DRIVER.index('\ndef lab(')],READER_DRIVER]
    path=HERE/'marlin_v9_reader.py';path.write_text('\n'.join(parts))
    print(path.relative_to(ROOT),hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':main()
