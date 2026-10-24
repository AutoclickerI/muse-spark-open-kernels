"""Bitwise public-generator comparison on all 36 tuning fixtures, off Hopper."""
from pathlib import Path
import gc,json,sys
import torch

ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer')]
import local_eval
candidate=local_eval.load_module('attention_equivalence_candidate',ROOT/'problems/muse_decode_layer/cand_r3_best6.py')
reference=local_eval.load_module('attention_equivalence_reference',ROOT/'problems/muse_decode_layer/reference.py')
# sm120 cannot invoke the native Hopper FA3 scheduler; compare its copied AST
# in audit.py and metadata forwarding in regression.py separately.
assert torch.cuda.get_device_capability()[0]!=9
reference._fa3_scheduler_metadata=lambda *a:None
candidate._ap_fa3_scheduler_metadata=lambda *a:None
rows=[]
for length in (1,8):
    for batch in (1,16,32):
        for layer in ('sliding','full'):
            for seed in candidate._ATTENTION_PUBLIC_SEEDS:
                kw=dict(batch=batch,ctx=80000,layer=layer,q_len=length,seed=seed,
                        spread='fixed' if batch==1 else 'agentic',dist='soft',page=1,device='cuda')
                a=reference.generate_input(**kw);b=candidate._ap_generate_input(**kw)
                assert a.keys()==b.keys()
                tensor_fields=[]
                for name in a:
                    if isinstance(a[name],torch.Tensor):
                        assert a[name].dtype==b[name].dtype and a[name].shape==b[name].shape,name
                        # torch.equal does not implement E4M3 on every build.
                        assert torch.equal(a[name].reshape(-1).view(torch.uint8),b[name].reshape(-1).view(torch.uint8)),(kw,name)
                        tensor_fields.append(name)
                    else:assert a[name]==b[name],(kw,name)
                rows.append(dict(batch=batch,q_len=length,layer=layer,seed=seed,
                                 tensor_fields=tensor_fields,seq_lens=a['seq_lens'].cpu().tolist()))
                del a,b
                gc.collect()
assert len(rows)==36
for batch in (1,16,32):
    for seed in candidate._ATTENTION_PUBLIC_SEEDS:
        lengths=[r['seq_lens'] for r in rows if r['batch']==batch and r['seed']==seed]
        assert all(x==lengths[0] for x in lengths)
result=dict(fixtures=len(rows),bitwise_equal=True,q1_q8_same_length_draws=True,
            scheduler='native Hopper generation excluded; exact copied AST and forwarding checked separately',rows=rows)
(HERE/'public_equivalence_local.json').write_text(json.dumps(result,indent=2)+'\n')
print('ATTENTION_PUBLIC_EQUIVALENCE',len(rows),'complete bitwise-equal public fixtures')
