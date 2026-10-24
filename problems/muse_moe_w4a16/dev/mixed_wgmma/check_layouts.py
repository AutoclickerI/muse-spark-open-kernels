"""Invert the canonical V6 and Humming/MMA+WGMMA adapters in PyTorch, then FP32 GEMM.

This tests stored layout and global-scale handling, not the Hopper device GEMM.
The Humming permutation follows image humming/kernel/process.cuh's repacker.
"""
import ast
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def unpack(q):return torch.stack((q&15,q>>4),-1).flatten(-2).long()


def decode(c,s):
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=c.device)
    return torch.where((c&8)!=0,-lut[c&7],lut[c&7])*s.float().repeat_interleave(16,-1)


def canonical():
    names={'_DQ_SLOTS','_LA_MASK','_bit_table','_TABLE','_encode_scales','_repack','repack_v6'}
    tree=ast.parse((ROOT/'problems/muse_prefill_layer/cand_mid_experts2.py').read_text())
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names) or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
    ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),'canonical','exec'),ns)
    return ns


def inverse_v6(bank,table,gated):
    e,t,_=bank.shape;r=t*16
    words=bank[:,:,:32768].contiguous().view(torch.int32).long().reshape(e,t,8,8,32,2,2)
    ct=torch.zeros((e,t,8,8,32,2,8,2),device=bank.device,dtype=torch.int64)
    for k,h,b,w,p in table:ct[...,k,h]|=((words[...,0 if w=='A' else 1]>>p)&1)<<b
    # [expert,tile,w,g,rr,q,p,slot,j,ri,half] -> original row,K.
    ct=ct.reshape(e,t,8,8,8,4,2,2,2,2,2)
    c=ct.permute(0,1,9,4,2,3,5,6,7,8,10).reshape(e,r,4096)
    u=bank[:,:,32768:].reshape(e,t,8,8,4,8,2).permute(0,1,6,3,2,5,4).reshape(e,r,256).long()
    sf=torch.where(u!=0,torch.exp2((u>>3).float()-21)*(1+(u&7).float()/8),0.)
    out=decode(c,sf)
    if gated:
        rows=torch.cat((torch.arange(t,device=bank.device)[:,None]*8+torch.arange(8,device=bank.device),
                        r//2+torch.arange(t,device=bank.device)[:,None]*8+torch.arange(8,device=bank.device)),1).flatten()
        ordered=torch.empty_like(out);ordered[:,rows]=out;out=ordered
    return out


def humming_indices(r,k,wgmma,dev):
    # For each source element in a 64x64 tile, record destination word and nibble.
    dest=torch.empty((64,64,3),dtype=torch.long)
    for lane in range(32):
        tmp=[None]*128
        for i in range(8):
            for j in range(8):
                for kk in range(2):
                    i4,i5=(j%2,i%2) if wgmma else (i%2,j%2)
                    ix=((((j//2)*4+i//2)*2+i4)*2+i5)*2+kk
                    tmp[ix]=(i*8+lane//4,j*8+(lane%4)*2+kk)
        for a in range(16):
            for nib in range(8):
                new=(nib%4)*2+nib//4
                row,col=tmp[a*8+new]
                dest[row,col]=torch.tensor((a//4,lane*4+a%4,nib))
    rr=torch.arange(r,device=dev)[:,None].expand(r,k);kk=torch.arange(k,device=dev)[None,:].expand(r,k)
    dd=dest.to(dev)[rr%64,kk%64]
    word=(kk//64*4+dd[...,0])*(r*2)+(rr//64)*128+dd[...,1]
    return word,dd[...,2]*4


def main():
    from humming import dtypes
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema
    from humming.transform import transform_humming_weight,transform_humming_weight_scale
    torch.manual_seed(49);dev='cuda';report={};canon=canonical()
    torch.backends.cuda.matmul.allow_tf32=False
    for gated in (False,True):
        e,r,k=3,128,4096
        q=torch.randint(256,(e,r,k//2),device=dev,dtype=torch.uint8)
        raw=torch.randint(256,(e,r,k//16),device=dev,dtype=torch.uint8)
        raw[(raw&127)==127]=126;s=raw.view(torch.float8_e4m3fn)
        gs=torch.tensor((.000013,.000041,.000007),device=dev)
        want=decode(unpack(q),s)
        back=inverse_v6(canon['repack_v6'](q,s,gated),canon['_TABLE'],gated)
        assert torch.equal(want,back)
        x=torch.randn((e,17,k),device=dev).bfloat16().float()
        a=torch.bmm(x,want.transpose(1,2))*gs[:,None,None]
        b=torch.bmm(x,back.transpose(1,2))*gs[:,None,None]
        assert torch.equal(a,b)
        report['canonical_gateup' if gated else 'canonical_down']={'decoded_weights_exact':True,'fp32_gemm_relative':0.,'global_scales':gs.tolist()}
    # Actual adapter schema and native repacker; independently invert both backend layouts.
    for wg in (False,True):
        e,r,k=3,128,128
        q=torch.randint(256,(e,r,k//2),device=dev,dtype=torch.uint8)
        raw=torch.randint(256,(e,r,k//16),device=dev,dtype=torch.uint8)
        raw[(raw&127)==127]=126;s=raw.view(torch.float8_e4m3fn)
        gs=torch.tensor((.000013,.000041,.000007),device=dev)
        schema,ts=ModeloptNvfp4WeightSchema().convert_humming(dict(weight=q,weight_scale=s,weight_scale_2=gs),
                    shape_n_stacks=[r//2,r//2],shape_k_stacks=[k],param_dtype=torch.bfloat16,num_experts=e)
        packed=transform_humming_weight(ts['weight'],dtypes.float4e2m1,dtypes.bfloat16,use_wgmma=wg,packed=True)
        word,shift=humming_indices(r,k,wg,dev)
        c=(packed.reshape(e,-1)[:,word]>>shift)&15
        assert torch.equal(c,unpack(q)),wg
        transformed=transform_humming_weight_scale(ts['weight_scale'])
        perm=torch.tensor([x+i for i in range(8) for x in (0,8,16,24,32,40,48,56)],device=dev)
        unscaled=transformed.view(torch.uint8).reshape(-1,64)[:,perm.argsort()].reshape(e,k//16,r).transpose(-1,-2).contiguous().view(torch.float8_e4m3fn)
        assert torch.equal(unscaled.view(torch.uint8),raw)
        assert torch.equal(ts['weight_scale_2'],gs)
        want=decode(unpack(q),s);back=decode(c,unscaled)
        x=torch.randn((e,17,k),device=dev).bfloat16().float()
        a=torch.bmm(x,want.transpose(1,2))*gs[:,None,None]
        b=torch.bmm(x,back.transpose(1,2))*ts['weight_scale_2'][:,None,None]
        assert torch.equal(a,b)
        report['humming_WGMMA' if wg else 'humming_MMA']={'codes_exact':True,'raw_scales_exact':True,'expert_global_scales_exact':True,'fp32_gemm_relative':0.}
    (HERE/'diagnostics/layout_agreement.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
