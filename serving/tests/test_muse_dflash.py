"""Exact DFlash glue and prepared dense GEMMs, fresh-input CUDA graphs, GPU0.

CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python \
 python serving/tests/test_muse_dflash.py --json runs/screen/serving_dflash/tests.json
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from sglang.srt.layers import muse_dflash as d
from sglang.srt.models.dflash import _grouped_conv, _score_edges

counts=Counter()
plans=[]


def stock_unary(x):
    return x.float().mul_(.1767766952966369).div_(20).tanh_().mul_(20)


def exact_replay(fn,ref,mutate,repeat=3):
    # Compare multiple independently replayed stock graphs too: this screen
    # observes zero stock spread, so no arbitrary error tolerance is added.
    with d._df_proj._Graph(fn) as a,d._df_proj._Graph(ref) as b:
        for _ in range(repeat):
            rng=torch.cuda.get_rng_state()
            mutate();a.graph.replay();b.graph.replay();torch.cuda.synchronize()
            assert torch.equal(a.out,b.out),(a.out.shape,float((a.out.float()-b.out.float()).abs().max()))
            saved=b.out.clone()
            torch.cuda.set_rng_state(rng);mutate();b.graph.replay()
            assert torch.equal(saved,b.out),'stock itself changed on identical replay inputs'
            counts['stock_repeat_checks']+=1
            counts['exact_graph_replays']+=1


class Tests(unittest.TestCase):
    def test_conv(self):
        for c in (1,4,16,32):
            for tapside in (0,1):
                t,h=c*8,8192
                x=torch.randn((t,h),device='cuda',dtype=torch.bfloat16)
                full=torch.randn((t,2,2,h//16),device='cuda',dtype=torch.bfloat16)
                delta=full[:,tapside]
                w=torch.randn((2,h),device='cuda',dtype=torch.bfloat16)
                ref=lambda:_grouped_conv(x,delta,w,8,h//16,16,2)
                exact_replay(lambda:d.dflash_conv2(x,delta,w),ref,
                             lambda:(x.copy_(torch.randn_like(x)),full.copy_(torch.randn_like(full)),w.copy_(torch.randn_like(w))),repeat=40)
                x.zero_();x[::8].fill_(1)
                self.assertTrue(torch.equal(d.dflash_conv2(x,delta,w),ref()))
                counts['conv_cases']+=1

    def test_unary(self):
        for c in (1,4,16,32):
            for dtype in (torch.bfloat16,torch.float32):
                x=torch.randn((7*c,16),device='cuda',dtype=dtype)*100
                stock_x=x.clone()
                def mutate():
                    fresh=torch.randn_like(x)*100;x.copy_(fresh);stock_x.copy_(fresh)
                exact_replay(lambda:d.dflash_unary(x),lambda:stock_unary(stock_x),mutate)
                counts['unary_cases']+=1

    def test_conv_wrapper(self):
        w=torch.randn((2048,8192),device='cuda',dtype=torch.bfloat16)/8192**.5
        base=torch.randn((2,2,8192),device='cuda',dtype=torch.bfloat16)
        op=d.DFlashConv2(w,base)
        for m in (8,32,128,256):
            x=torch.randn((m,8192),device='cuda',dtype=torch.bfloat16)
            def ref():
                coef=F.linear(x,w).reshape(m,2,2,512)
                first=_grouped_conv(x,coef[:,0],base[0],8,512,16,2)
                return _grouped_conv(first,coef[:,1],base[1],8,512,16,2)
            def fast():
                first,coef=op.prepare(x)
                return op.finish(first,coef)
            exact_replay(fast,ref,lambda:x.copy_(torch.randn_like(x)))
            counts['complete_conv_wrappers']+=1

    def test_selector(self):
        for c in (1,4,16,32):
            v,r,l,k=4096,256,7,16
            a=torch.randn((v,r),device='cuda',dtype=torch.bfloat16)
            b=torch.randn_like(a);ids=torch.randint(v,(c,l,k),device='cuda')
            h=torch.randn((c,l,r),device='cuda',dtype=torch.bfloat16)
            u=torch.randn((c,l,k),device='cuda');anchor=torch.randint(v,(c,),device='cuda')
            ref=lambda:_score_edges(predecessor_table=a,successor_table=b,candidate_ids=ids,
                                    unary_logits=u,hidden=h,anchor_token_ids=anchor,top_k=k)
            exact_replay(lambda:d.dflash_selector_scores(a,b,ids,u,h,anchor),ref,
                         lambda:(ids.random_(0,v),h.copy_(torch.randn_like(h)),u.normal_()))
            counts['selector_cases']+=1

    def test_dense(self):
        # All genuine per-rank dense shapes, including seven prediction slots.
        specs=[('qkv',1280,8192,8),('o',8192,1024,8),('gate_up',7168,8192,8),
               ('down',8192,3584,8),('conv_coeff',2048,8192,8),
               ('context_fc',1024,40960,8),('context_kv',1280,8192,8),
               ('draft_head',25256,8192,7)]
        for name,n,k,mult in specs:
            w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)/k**.5
            ms=tuple(c*mult for c in (1,4,16,32))
            op=d.DFlashLinear(w.T.contiguous() if name=='context_kv' else w,
                              tokens=ms,budget_s=12,stock_nn=name=='context_kv')
            plans.append(dict(name=name,shape=[n,k],plans={str(m):p for m,p in op.plans.items()}))
            for m in ms:
                x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
                exact_replay(lambda:op(x),lambda:op._df_stock(x),lambda:x.copy_(torch.randn_like(x)))
                counts['dense_cases']+=1
            del op,w

    def test_capture_and_guards(self):
        self.assertEqual(d._df_proj._Graph.active_streams,0)
        w=torch.randn((1280,8192),device='cuda',dtype=torch.bfloat16)
        op=d.DFlashLinear(w,tokens=())
        op.prepare(tokens=(8,),budget_s=0)
        w.add_(1)
        with self.assertRaisesRegex(RuntimeError,'weight changed'):
            op(torch.zeros((8,8192),device='cuda',dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError,'2D'):
            op(torch.zeros((4,7,8192),device='cuda',dtype=torch.bfloat16))
        self.assertEqual(d._df_proj._Graph.active_streams,0)

    def test_hopper_dense_glue(self):
        native=d._df_proj._module()
        class Portable:
            def options(self,m,n,k,nn):return native.options(m,n,k,nn)
            def skinny(self,x,w,out,config):
                nt,tr,u,wn,wk,ks,xp=config
                assert x.shape[1]%(wk*32*u*ks)==0
                out.copy_(F.linear(x,w))
            def lt(self,x,w,out,work,codes,index,nn):
                assert 0<=index<codes.shape[0] and work.numel()==32<<20
                out.copy_(torch.mm(x,w) if nn else F.linear(x,w))
        w=torch.randn((1280,8192),device='cuda',dtype=torch.bfloat16)
        x=torch.randn((16,8192),device='cuda',dtype=torch.bfloat16)
        with patch.object(torch.cuda,'get_device_capability',return_value=(9,0)),patch.object(d._df_proj,'_module',return_value=Portable()):
            for nn in (False,True):
                op=d.DFlashLinear(w.T.contiguous() if nn else w,tokens=(),stock_nn=nn)
                for config in [('stock',)]+[('skin',s) for s in (1,2,4,8)]+[('lt',0),('lt_nn',0)]:
                    op._df_ensure(16,config);op.plans[16]=config
                    expected=torch.mm(x,op.transpose) if config[0]=='lt_nn' else op._df_stock(x) if config[0]=='stock' else F.linear(x,op.weight)
                    self.assertTrue(torch.equal(op(x),expected))
                    counts['forced_hopper_dense_cases']+=1



if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path);args=ap.parse_args()
    torch.manual_seed(10026024)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report=dict(ok=result.wasSuccessful(),tests=result.testsRun,cc=torch.cuda.get_device_capability(),
                sha256=hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest(),
                counts=dict(counts),plans=plans,agreement='bit exact; observed stock spread zero')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));raise SystemExit(0 if result.wasSuccessful() else 1)
