import time
import statistics

GB_TOKENS=(1,8,16,32,128,256)


def gb_choices(m,phase):
    for bt in tuple(dict.fromkeys((min(m,8),min(m,16),min(m,4)))):
        for bn in (64,32):
            for splits in ((2,1,4) if phase=='b1' else (8,4,2) if phase=='pool' else (8,4,2,1)):
                yield bt,bn,splits,64 if phase=='pool' else 128,4


class GbPlan:
    def __init__(self,state,m,phase,choice):
        self.m,self.phase,self.choice=m,phase,choice
        bt,bn,splits,bk,warps=choice
        if phase not in ('b1','pool','b2') or bt<1 or bt>32 or bn not in (32,64) or splits not in (1,2,4,8,16):
            raise ValueError('invalid glue plan')
        if bk not in (32,64,128) or warps not in (4,8):raise ValueError('invalid glue launch')
        device=state['o'].device
        self.n=4608 if phase=='pool' else 8192 if phase=='b1' or m>8 else 1024
        self.k=8192 if phase=='pool' else 1024 if phase=='b1' else 4096
        if self.k%splits or (self.k//splits)%bk:raise ValueError('invalid split granule')
        self.part=torch.empty((splits,m,4096 if phase=='pool' else self.n),device=device,dtype=torch.float32)
        self.rpart=torch.empty((splits,m,256),device=device,dtype=torch.float32) if phase=='pool' else self.part
        # Only synchronization state is prepared. All input-dependent outputs are overwritten each call.
        self.counter=torch.zeros((triton.cdiv(m,bt)*(2 if phase=='pool' else 1),),device=device,dtype=torch.int32)
        self.state=state

    def run(self,x,stream=None,data=None):
        st=self.state;m=self.m;bt,bn,splits,bk,warps=self.choice;tb=max(16,triton.next_power_of_2(bt))
        if x.shape!=(m,self.k) or x.dtype!=torch.bfloat16 or not x.is_contiguous():raise ValueError('glue input layout')
        mt=triton.cdiv(m,bt);dev=x.device
        if self.phase=='pool':
            if data is None:raise ValueError('routing requires local_start')
            y=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
            ids=torch.empty((m,8),device=dev,dtype=torch.int32)
            weights=torch.empty((m,8),device=dev,dtype=torch.float32);lids=torch.empty_like(ids)
            _gb_pool_route[((4096//bn+1)*mt,splits)](x,st['pool_router'],st['pre_expert_w'],
                st['router_bias'],data['local_start'],self.part,self.rpart,self.counter,y,ids,weights,lids,
                M=m,BN=bn,BK=bk,KS=8192//splits,BT=bt,TB=tb,WARPS=warps,
                num_warps=warps,num_stages=2,enable_fp_fusion=False)
            return y,ids,weights,lids
        if stream is None or stream.shape!=(m,8192) or stream.dtype!=torch.float32:raise ValueError('glue stream layout')
        b2=self.phase=='b2';name=('unpooler_slice' if m<=8 else 'unpooler') if b2 else 'o'
        a,b,pw=('alpha2','beta2','next_pre_attn_w') if b2 else ('alpha1','beta1','pre_ffn_w')
        out=torch.empty((m,8192),device=dev,dtype=torch.bfloat16);sout=torch.empty_like(stream)
        _gb_boundary_gemv[(self.n//bn*mt,splits)](x,st[name],stream,st[a],st[b],st['post_ffn_w'],st[pw],
            self.part,self.counter,out,sout,M=m,N=self.n,K=self.k,BN=bn,BK=bk,KS=self.k//splits,
            BT=bt,TB=tb,POSTW=b2,WARPS=warps,num_warps=warps,num_stages=2,enable_fp_fusion=False)
        return out,sout
