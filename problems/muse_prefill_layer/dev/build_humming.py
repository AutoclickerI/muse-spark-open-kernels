from pathlib import Path
p=Path('problems/muse_prefill_layer')
s=(p/'cand_v2.py').read_text()
other=Path('problems/muse_moe_w4a16/cand_humming.py').read_text()
imports=other[other.index('import json'):other.index('class Weights')]
cls=other[other.index('class _Experts:'):other.index('def prepare(weights: Weights):')].replace('weights: Weights','weights')
cls=cls.replace('self.compile(range(1, PRECOMPILED_MAX_BS + 1))', 'self.compile([1 << i for i in range(16)])')
cls=cls.replace('    def plan(self, bs: int):\n        if bs not in self.plans:  # beyond PRECOMPILED_MAX_BS: compiles here, on the first (eager) call\n            self.compile([bs])\n        return self.plans[bs]', '    def plan(self, bs: int):\n        bucket = 1 << (max(bs, 1) - 1).bit_length()\n        if bucket not in self.plans:\n            self.compile([bucket])\n        return self.routes(bs), self.plans[bucket][1]')
const='\nPRECOMPILED_MAX_BS = 32768\nCOMPUTE_CONFIG = json.dumps({"use_f16_accum": False, "gemm_type": GemmType.INDEXED.value})\n'
s=s.replace('def prepare(weights):','def prepare_base(weights):')
start=s.index('    if T >= BIG_TOKENS or st["bank"] is None:')
end=s.index('    if T <= 8:',start)
s=s[:start]+'''    moe = _experts_humming(pooled, topk_w, slot_e, st)
'''+s[end:]
s+='\n\n'+imports+'\n'+const+'\n'+cls+'''

def prepare(weights):
    st = prepare_base(weights)
    st['humming'] = _Experts(weights, 0, 32)
    return st


def _experts_humming(pooled, topk_w, slot_e, st):
    T = pooled.shape[0]
    x = torch.empty_like(pooled)
    _pre_expert_norm_kernel[(T,)](pooled, st['pre_expert'], x, D=D_EXPERT, num_warps=4)
    bank = st['humming']
    routes, gemms = bank.plan(T)
    local_ids = slot_e
    gate_up = torch.empty((T*8,8192),dtype=x.dtype,device=x.device)
    act = torch.empty((T*8,4096),dtype=x.dtype,device=x.device)
    y = torch.empty_like(act)
    aligned = {}
    for name, inputs, outputs, top_k in (('w13',x,gate_up,8),('w2',act,y,1)):
        block_m, launch = gemms[name]
        if block_m not in aligned:
            aligned[block_m] = moe_align_block_size(local_ids,block_m,bank.count,ignore_invalid_expert=True)
        sorted_ids,block_experts,n_padded = aligned[block_m]
        humming_ops.launch_kernel(configs=launch,inputs=inputs,outputs=outputs,**bank.weight(name),
                locks=bank.locks,sorted_ids=sorted_ids,expert_ids=block_experts,
                num_tokens_padded=n_padded,top_k=top_k,valid_shape_m=routes)
        if name == 'w13':
            silu_and_mul(gate_up,act,expert_ids=local_ids.view(-1),expert_step=1)
    return fused_post_expert_norm_weighted_sum(y.view(T,8,4096),topk_w,
            norm_weight=st['post_expert_bf16'],eps=POST_EPS,weight_first=True,slot_expert_ids=local_ids)
'''
(p/'cand_v6.py').write_text(s)
