"""Exercise all receipt-table branches on the real full-layer local graphs."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev/best6')]
import local_eval
import force as best6_force


def install(m,cold,exercise_table,exercise_prefetch=False):
    records=best6_force.install(m,cold)
    if exercise_table:
        for key,row in list(m.BEST7_SETTINGS.items()):
            if key=='other':continue
            batch,length,window=key;tokens=batch*length
            glue='best5' if window else ('best6',(16,4 if tokens<=8 else 1),(8,1),(8,16 if tokens<=8 else 1))
            attention=('fa3',(16,True)) if length==8 else (('layer',None) if window else ('attn_main',None))
            moe='v9nt2' if tokens<=32 else row[2]
            config=0 if tokens==1 else 10 if tokens<=32 else 20
            names=('qkv_gate','o','pool_router','unpooler_slice' if tokens<=8 else 'unpooler')
            projection={name:('stream',config,4) for name in names}
            if exercise_prefetch and length==1:projection['l2_prefetch']=10 if window else 11
            m.BEST7_SETTINGS[key]=(glue,attention,moe,projection)
    original=m.prepare
    def prepare(weights):
        state=original(weights)
        # The real Humming Python transform executes here even with a launch
        # stand-in. It must only permute the original stored scale bytes.
        records['humming_raw_scales']={}
        bank=state['moe6_adapters']['humming']
        for name in ('w13','w2'):
            got=bank.weight(name)['weight_scale'].view(m.torch.uint8).reshape(-1)
            want=weights[name+'_sf'].view(m.torch.uint8).reshape(-1)
            assert got.numel()==want.numel(),(name,got.shape,want.shape)
            gh=m.torch.bincount(got.long(),minlength=256)
            wh=m.torch.bincount(want.long(),minlength=256)
            assert m.torch.equal(gh,wh),name+' Humming changed E4M3 scale bytes'
            records['humming_raw_scales'][name]=True
        assert len(state['best7_plans'])==13
        if exercise_table:
            assert len(state['best7_stream_ops'])==24
            assert {p['glue'] for k,p in state['best7_plans'].items() if k!='other'}=={'best5','best6'}
            for key,plan in state['best7_plans'].items():
                if key!='other' and key[0]*key[1]<=32:assert plan['moe']=='v9nt2'
        if exercise_prefetch:assert len(state['best7_prefetch_ops'])==6
        assert len(state['attention_public_rows'])==12
        records['plans']={str(k):v for k,v in state['best7_plans'].items()}
        records['gates']={str(k):v for k,v in state['best7_gates'].items()}
        records['stream_ops']=len(state['best7_stream_ops'])
        records['prefetch_ops']=len(state['best7_prefetch_ops'])
        choices={m:{n:0 for n in ('raw17','humming_masked','dg_w8')} for m in (128,256)}
        original_expert=m._best7_expert;counter={'calls':0}
        def expert(pr,x,wts,lids,st,plan):
            if x.shape[0] in choices:
                name=('raw17','humming_masked','dg_w8')[counter['calls']%3]
                counter['calls']+=1;choices[x.shape[0]][name]+=1
                return original_expert(pr,x,wts,lids,st,dict(plan,moe=name))
            return original_expert(pr,x,wts,lids,st,plan)
        if exercise_table:m._best7_expert=expert
        records['full_layer_experts']=choices
        print('BEST7_FORCED_READY',json.dumps(dict(stream_ops=records['stream_ops'],gates=len(records['gates']))),flush=True)
        return state
    m.prepare=prepare
    return records
