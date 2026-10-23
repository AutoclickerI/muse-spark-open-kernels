"""Muse TP8 local projection admission; GPU0, native sm_120 and forced Hopper.

CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python \
  python serving/tests/test_muse_proj.py --json runs/screen/serving_proj/tests.json
Use --build-only 9.0a to cross-compile the complete native module for Hopper.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import logging
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from sglang.srt.layers import muse_proj as p


COUNTS=Counter()
WORST=0
PLANS=[]


def agree(a,b):
    global WORST
    error=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
    WORST=max(error,WORST)
    assert p._agrees(a,b),(a.shape,error)
    return error


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(710241)
        cls.ops=[]
        cls.started=time.monotonic()
        for role,shapes in p._SHAPES.items():
            for shape in shapes:
                w=torch.randn(shape,device='cuda',dtype=torch.bfloat16)/shape[1]**.5
                op=p.MuseProjection(w,role,tokens=(8,16,32,64,128,256))
                cls.ops.append(op)
                PLANS.append(dict(role=role,shape=shape,plans={str(k):v for k,v in op.plans.items()}))
        assert p._Graph.active_streams==0,'prepare left live tuning streams'

    def test_00_model_tp8_dimensions(self):
        from sglang.srt.configs.muse_spark import MuseSpark1xTextConfig
        cfg=MuseSpark1xTextConfig();tp=8
        self.assertEqual(p._SHAPES['o'],((cfg.hidden_size,cfg.num_attention_heads*cfg.head_dim//tp),))
        self.assertEqual(set(p._SHAPES['pooler']),{(cfg.moe_hidden_size,cfg.hidden_size),(cfg.moe_hidden_size//tp,cfg.hidden_size)})
        self.assertEqual(set(p._SHAPES['unpooler']),{(cfg.hidden_size,cfg.moe_hidden_size),(cfg.hidden_size//tp,cfg.moe_hidden_size)})
        COUNTS['model_tp8_shape_checks']+=5

    def test_01_all_local_shapes_and_nearby_tokens(self):
        for op in self.ops:
            for m in (0,1,3,8,9,16,24,32,64,128,256,264,753):
                with self.subTest(role=op.kind,n=op.weight.shape[0],m=m):
                    x=torch.randn((m,op.weight.shape[1]),device='cuda',dtype=torch.bfloat16)
                    got=op(x);ref=F.linear(x,op.weight)
                    self.assertEqual(got.dtype,torch.bfloat16)
                    if m:agree(got,ref)
                    else:self.assertEqual(got.shape,ref.shape)
                    COUNTS['public_shapes']+=1

    def test_02_all_native_variants_fresh_graphs(self):
        for op in self.ops:
            for m in (8,16,32,64,128,256):
                x=torch.randn((m,op.weight.shape[1]),device='cuda',dtype=torch.bfloat16)
                variants=list(dict.fromkeys(op._variants(m)))
                for config in variants:
                    with self.subTest(role=op.kind,n=op.weight.shape[0],m=m,variant=config):
                        op._ensure(m,config)
                        g=p._Graph(lambda:op._run(x,config))
                        for seed in range(3):
                            fresh=torch.randn_like(x)
                            if seed==1:fresh[:,::257]*=32
                            if seed==2:fresh.zero_();fresh[:,127]=.5
                            x.copy_(fresh);g.graph.replay()
                            agree(g.out,F.linear(x,op.weight))
                            if seed==2:
                                expected=(op.weight[:,127]*.5).expand(m,-1)
                                self.assertTrue(torch.equal(g.out,expected),(op.kind,m,config,'weight-column probe'))
                            COUNTS['native_fresh_replays']+=1
                        COUNTS['native_'+config[0]]+=1
                        g.close()
                # Native tests can create a transient NN layout after prepare.
                # Retain it while forced graph tests below override plans.

    def test_03_public_graph_and_concurrent_workspaces(self):
        for op in self.ops:
            m=128
            # Force both cuBLASLt layouts, including a per-call workspace on
            # two streams with different activation buffers and graph pools.
            for nn in (0,1):
                config=('lt_nn' if nn else 'lt',0)
                op._ensure(m,config);op.plans[m]=config
                xs=[torch.randn((m,op.weight.shape[1]),device='cuda',dtype=torch.bfloat16) for _ in range(2)]
                graphs=[p._Graph(lambda x=x:op(x)) for x in xs]
                for seed in range(3):
                    for index in range(2):
                        xs[index].copy_(torch.randn_like(xs[index]))
                    for graph in graphs:graph.stream.wait_stream(torch.cuda.current_stream())
                    for index in (0,1,1,0):
                        with torch.cuda.stream(graphs[index].stream):graphs[index].graph.replay()
                    torch.cuda.synchronize()
                    for x,g in zip(xs,graphs):agree(g.out,F.linear(x,op.weight))
                    COUNTS['concurrent_graph_pairs']+=1
                for graph in graphs:graph.close()
            m=8;key=(op.weight.device.index,op.kind,tuple(op.weight.shape),m)
            op.plans[m]=p._PLAN_CACHE.get(key,op._base(m))
            x=torch.randn((m,op.weight.shape[1]),device='cuda',dtype=torch.bfloat16)
            graph=p._Graph(lambda:op(x))
            for _ in range(3):
                x.copy_(torch.randn_like(x));graph.graph.replay();agree(graph.out,F.linear(x,op.weight))
                COUNTS['public_graph_replays']+=1
            graph.close()

    def test_04_rank_slices_and_collective_boundary(self):
        for role,shape in (('pooler',(4096,8192)),('unpooler',(8192,4096))):
            w=torch.randn(shape,device='cuda',dtype=torch.bfloat16)/shape[1]**.5
            for m in (8,32,128):
                ops=[p.MuseProjection(w.chunk(8)[rank],role,tokens=(m,)) for rank in range(8)]
                x=torch.randn((m,shape[1]),device='cuda',dtype=torch.bfloat16)
                graph=p._Graph(lambda:torch.cat([op(x) for op in ops],dim=1))
                for _ in range(3):
                    x.copy_(torch.randn_like(x));graph.graph.replay()
                    agree(graph.out,F.linear(x,w))
                    COUNTS['tp8_gather_replays']+=1
                graph.close()
        # O shards along K; local outputs must still be reduced by the caller.
        weights=[torch.randn((8192,1024),device='cuda',dtype=torch.bfloat16)/1024**.5 for _ in range(8)]
        ops=[p.MuseProjection(w,'o',tokens=(8,)) for w in weights]
        xs=[torch.randn((8,1024),device='cuda',dtype=torch.bfloat16) for _ in range(8)]
        graph=p._Graph(lambda:torch.stack([op(x).float() for op,x in zip(ops,xs)]).sum(0))
        for _ in range(3):
            for x in xs:x.copy_(torch.randn_like(x))
            graph.graph.replay()
            expected=torch.stack([F.linear(x,w).float() for x,w in zip(xs,weights)]).sum(0)
            agree(graph.out,expected);COUNTS['tp8_o_reduce_replays']+=1
        graph.close()

    def test_05_rejects_bad_trials_and_stops_budget(self):
        w=torch.randn((8192,1024),device='cuda',dtype=torch.bfloat16)
        op=p.MuseProjection(w,'o',tokens=())
        native=op._run
        def run(x,config):
            if config==('broken',):raise ValueError('injected launch failure')
            if config==('wrong',):return torch.zeros((x.shape[0],8192),device=x.device,dtype=x.dtype)
            return native(x,config)
        with patch.object(op,'_run',side_effect=run),patch.object(op,'_variants',return_value=iter((('cublas',),('broken',),('wrong',)))):
            with self.assertLogs(p.logger,logging.WARNING) as logs:op.prepare((129,))
            self.assertTrue(any('injected launch failure' in s for s in logs.output))
            self.assertTrue(any('1% torch guard' in s for s in logs.output))
        self.assertEqual(op.plans[129],('cublas',))
        op.prepare((31,33,255),budget_s=0)
        x=torch.randn((255,1024),device='cuda',dtype=torch.bfloat16)
        agree(op(x),F.linear(x,w))
        # Later layers must still reuse validated shape choices after the
        # new-trial budget expires, rather than silently reverting to cuBLAS.
        key=(op.weight.device.index,op.kind,tuple(op.weight.shape),128)
        cached=p._PLAN_CACHE[key]
        with patch.object(op,'_variants',side_effect=AssertionError('trial after deadline')):
            op.prepare((128,),budget_s=0)
        self.assertEqual(op.plans[128],cached)
        COUNTS['injected_rejections']+=2

    def test_06_immutable_weights_and_capture_contract(self):
        w=torch.randn((8192,1024),device='cuda',dtype=torch.bfloat16)
        op=p.MuseProjection(w,'o',tokens=())
        w.add_(.125)
        with self.assertRaisesRegex(RuntimeError,'weight changed'):op(torch.zeros((8,1024),device='cuda',dtype=torch.bfloat16))
        graph=torch.cuda.CUDAGraph();x=torch.zeros((8,1024),device='cuda',dtype=torch.bfloat16)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        self.assertEqual(p._Graph.active_streams,0)
        with torch.cuda.graph(graph,stream=stream):
            gc.collect()  # closed tuning graphs must require no CUDA cleanup
            with self.assertRaisesRegex(RuntimeError,'precede graph'):p.MuseProjection(w,'o',tokens=())
            with self.assertRaisesRegex(RuntimeError,'before graph'):op.prepare(())
            y=x+1
        graph.replay();torch.cuda.synchronize();self.assertTrue(bool((y==1).all()))
        graph.reset()
        with torch.inference_mode():
            inf_w=torch.randn((8192,1024),device='cuda',dtype=torch.bfloat16)
            inf_op=p.MuseProjection(inf_w,'o',tokens=(8,))
            agree(inf_op(x),F.linear(x,inf_w))
        COUNTS['capture_and_mutation_contracts']+=4

    def test_07_force_hopper_gate_and_native_abi_standin(self):
        native=p._module()
        class Portable:
            def __init__(self):self.skin_calls=0;self.lt_calls=0
            def options(self,m,n,k,nn):
                codes=native.options(m,n,k,nn)
                return codes if codes.shape[0]>=2 else codes[:1].repeat(2,1)
            def skinny(self,x,weight,out,config):
                nt,tr,u,wn,wk,ks,xp=config
                assert nt*8>=x.shape[0] and u==4 and ks in (1,2,4,8)
                assert x.shape[1]%(ks*wk*32*u)==0
                out.copy_(F.linear(x,weight));self.skin_calls+=1
            def lt(self,x,weight,out,workspace,codes,index,nn):
                assert workspace.numel()==32<<20 and index<codes.shape[0]
                out.copy_(x@weight if nn else F.linear(x,weight));self.lt_calls+=1
        portable=Portable()
        with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)),patch.object(p,'_module',return_value=portable):
            for original in self.ops:
                op=p.MuseProjection(original.weight,original.kind,tokens=())
                for m in (8,16,32,128,256):
                    x=torch.randn((m,op.weight.shape[1]),device='cuda',dtype=torch.bfloat16)
                    self.assertTrue(p.projection_supported(x,op.weight,op.kind))
                    self.assertFalse(p.projection_supported(x[:,::2],op.weight,op.kind))
                    config=p._RECEIPT.get((op.kind,m),op._base(m)) if op._full_shape() else op._base(m)
                    configs=[config]
                    if not op._full_shape() and m<=16 and not (op.kind=='unpooler' and m<=8):
                        configs.extend(cfg for cfg in op._variants(m) if cfg[0]=='skin' and len(cfg)>1)
                    for config in configs:
                        op._ensure(m,config);op.plans[m]=config
                        graph=p._Graph(lambda:op(x))
                        x.copy_(torch.randn_like(x));graph.graph.replay();agree(graph.out,F.linear(x,op.weight))
                        graph.close()
                        COUNTS['forced_hopper_cases']+=1
        self.assertGreater(portable.skin_calls,0);self.assertGreater(portable.lt_calls,0)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-only',choices=('9.0a','12.0'))
    ap.add_argument('--json',type=Path)
    args=ap.parse_args()
    if args.build_only:
        p._module(args.build_only)
        print(json.dumps(dict(arch=args.build_only,build='ok',
                              module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
                              cuda_sha256=hashlib.sha256(p._CUDA_SOURCE.encode()).hexdigest())))
        raise SystemExit(0)
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProjectionTests))
    report=dict(ok=result.wasSuccessful(),tests=result.testsRun,gpu=torch.cuda.get_device_name(),
                cc=torch.cuda.get_device_capability(),seconds=time.monotonic()-started,
                module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
                worst_relative_l2=WORST,counts=dict(COUNTS),plans=PLANS)
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
