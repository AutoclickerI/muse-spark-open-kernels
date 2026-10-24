"""Force Hopper glue and require native S6 math at every scored dispatch row."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
sys.path.insert(0,str(R/'tools'))
import local_eval
spec=importlib.util.spec_from_file_location('best7_force',H.parent/'best7/force.py')
best7_force=importlib.util.module_from_spec(spec);spec.loader.exec_module(best7_force)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');args=ap.parse_args()
    candidate=R/'problems/muse_decode_layer/cand_r3_best7_s6.py'
    original=local_eval.load_module;records={};context={'row':None};projection_rows={}
    counters={n:dict(calls=0,captured_calls=0,full_layer_rows={}) for n in ('nt1_lop3','hh_m128','mixed_parallel_m256')}
    def loader(name,path):
        m=original(name,path)
        if name!='submission':return m
        records['forced']=best7_force.install(m,args.cold_native,False)
        def counted(variant):
            counter=counters[variant];capture=m.torch.cuda.is_current_stream_capturing()
            counter['calls']+=1;counter['captured_calls']+=int(capture)
            if context['row'] is not None:
                row=counter['full_layer_rows'].setdefault(context['row'],dict(eager=0,captured=0))
                row['captured' if capture else 'eager']+=1
        nt=m.nt1_call;dist=m.pm_call;rec=m.wi_call
        def small(data,bank,kind='recode',stages=2,grid=264,trace=False):
            assert (kind,stages,grid,trace)==('lop3',2,264,False)
            assert data['hidden_states'].shape[0] in (1,8,16,32)
            counted('nt1_lop3');return nt(data,bank,kind,stages,grid,trace)
        def large(data,base,word,native,first,second):
            assert (first,second)==('H','H')
            assert data['hidden_states'].shape[0]==128
            assert all(word[k].dtype==m.torch.int16 for k in ('w13_sf','w2_sf'))
            counted('hh_m128');return dist(data,base,word,native,first,second)
        def recoded(data,bank,policy=-1,grid=2112):
            assert (policy,grid)==(6,2112)
            assert all(k in bank for k in ('w13_rec','w13_rec_sf','w2_rec','w2_rec_sf','mix_stream','mix_start','mix_done'))
            counted('mixed_parallel_m256');return rec(data,bank,policy,grid)
        m.nt1_call=small;m.pm_call=large;m.wi_call=recoded
        run=m._best7_run
        def layer(data,state,plan,prefetch=None):
            c=data['config'];prior=context['row']
            key=(c['batch'],c['q_len'],c['window'])
            if key in m.BEST7_SETTINGS:
                expected='nt1_lop3' if c['tokens']<=32 else 'hh_m128' if c['tokens']==128 else 'mixed_parallel_m256'
                assert plan['moe']==expected,(key,plan)
                context['row']=str(key)
            try:return run(data,state,plan,prefetch)
            finally:context['row']=prior
        m._best7_run=layer
        project=m._best7_project
        def projected(x,name,state,plan,out_dtype=m.torch.bfloat16):
            if x.shape[0]==1 and name=='o' and plan['projection'].get('o')==('stream',6,1):
                assert context['row'] in ('(1, 1, 2048)','(1, 1, 0)'),context
                row=projection_rows.setdefault(context['row'],dict(eager=0,captured=0))
                row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
            return project(x,name,state,plan,out_dtype)
        m._best7_project=projected
        prepared=m.prepare
        def prepare(weights):
            state=prepared(weights);comparisons=[]
            for window,kind in ((0,'full'),(2048,'sliding')):
                fixtures=[m._ap_generate_input(batch=1,ctx=80000,layer=kind,q_len=1,
                    dist='soft',page=1,seed=seed,device=weights['o'].device) for seed in (172903,180822,188741)]
                plan=state['best7_plans'][1,1,window]
                assert plan['projection']=={'o':('stream',6,1)},plan
                base=dict(plan,projection={})
                graphs=[m._best5_graph(lambda data,p=p:m._best7_run(data,state,p),fixtures[0]) for p in (base,plan)]
                for index,fixture in enumerate(fixtures):
                    for graph in graphs:local_eval.copy_into(graph[0],fixture);graph[1].replay()
                    m.torch.cuda.synchronize()
                    errors=[]
                    for got,want in zip(graphs[1][2][:2],graphs[0][2][:2]):
                        error,row=m._best5_error(got,want);assert error<=.003 and row<=.015,(kind,index,error,row)
                        errors.append([error,row])
                    assert m.torch.equal(graphs[1][2][2],graphs[0][2][2]),('M1 O route IDs changed',kind,index)
                    comparisons.append(dict(layer=kind,seed=(172903,180822,188741)[index],errors=errors,route_ids_exact=True))
                for graph in graphs:graph[1].reset()
            records['projection_live_comparisons']=comparisons
            return state
        m.prepare=prepare
        return m
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(candidate),'--no-baseline','--json',str(H/'forced_local.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:
            expected={v:set() for v in counters}
            for key,plan in records['forced']['plans'].items():
                if key=='other':continue
                assert plan['moe'] in counters,(key,plan)
                expected[plan['moe']].add(key)
            for variant,counter in counters.items():
                assert set(counter['full_layer_rows'])==expected[variant],(variant,counter,expected)
                assert all(row['captured']>0 and row['eager']>0 for row in counter['full_layer_rows'].values())
            assert [len(expected[n]) for n in counters]==[8,2,2]
            assert set(projection_rows)=={'(1, 1, 2048)','(1, 1, 0)'}
            assert all(v['captured']>0 and v['eager']>0 for v in projection_rows.values())
            records.update(projection_native_rows=projection_rows,sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),native_sm120=counters,
                           limit='Hopper-only attention/DG/Humming launches use established stand-ins; all three shipped expert engines run actual portable CUDA math')
            (H/'forced_dispatch_local.json').write_text(json.dumps(records,indent=2)+'\n')
        raise

if __name__=='__main__':main()
