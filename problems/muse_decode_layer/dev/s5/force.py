"""Force Hopper glue and require native S5 math at every scored dispatch row."""
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
    candidate=R/'problems/muse_decode_layer/cand_r3_best7_s5.py'
    original=local_eval.load_module;records={};context={'row':None}
    counters={n:dict(calls=0,captured_calls=0,full_layer_rows={}) for n in ('nt1_lop3','hh_m128','recoded_c1g2112')}
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
        nt=m.nt1_call;dist=m.pm_call;rec=m.rc_call
        def small(data,bank,kind='recode',stages=2,grid=264,trace=False):
            assert (kind,stages,grid,trace)==('lop3',2,264,False)
            assert data['hidden_states'].shape[0] in (1,8,16,32)
            counted('nt1_lop3');return nt(data,bank,kind,stages,grid,trace)
        def large(data,base,word,native,first,second):
            assert (first,second)==('H','H')
            assert data['hidden_states'].shape[0]==128
            assert all(word[k].dtype==m.torch.int16 for k in ('w13_sf','w2_sf'))
            counted('hh_m128');return dist(data,base,word,native,first,second)
        def recoded(data,bank,which=0,grid=1056):
            assert (which,grid,bank['encoded'])==(1,2112,True)
            counted('recoded_c1g2112');return rec(data,bank,which,grid)
        m.nt1_call=small;m.pm_call=large;m.rc_call=recoded
        run=m._best7_run
        def layer(data,state,plan,prefetch=None):
            c=data['config'];prior=context['row']
            key=(c['batch'],c['q_len'],c['window'])
            if key in m.BEST7_SETTINGS:
                expected='nt1_lop3' if c['tokens']<=32 else 'hh_m128' if c['tokens']==128 else 'recoded_c1g2112'
                assert plan['moe']==expected,(key,plan)
                context['row']=str(key)
            try:return run(data,state,plan,prefetch)
            finally:context['row']=prior
        m._best7_run=layer
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
            records.update(sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),native_sm120=counters,
                           limit='Hopper-only attention/DG/Humming launches use established stand-ins; all three shipped expert engines run actual portable CUDA math')
            (H/'forced_dispatch_local.json').write_text(json.dumps(records,indent=2)+'\n')
        raise

if __name__=='__main__':main()
