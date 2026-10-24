"""Exercise S2's shipped table and Hopper glue; the new expert runs native sm120 math."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
sys.path.insert(0,str(R/'tools'))
import local_eval
spec=importlib.util.spec_from_file_location('best7_force',H.parent/'best7/force.py')
best7_force=importlib.util.module_from_spec(spec);spec.loader.exec_module(best7_force)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');a=ap.parse_args()
    candidate=R/'problems/muse_decode_layer/cand_r3_best7_scale_w2fix.py'
    original=local_eval.load_module;records={};counter={'calls':0,'captured_calls':0,'full_layer_rows':{}};context={'row':None}
    def loader(name,path):
        m=original(name,path)
        if name!='submission':return m
        records['forced']=best7_force.install(m,a.cold_native,False)
        call=m.fx_call
        def counted(data,bank,which=0,grid=528,mode=1):
            assert (which,grid,mode)==(1,2112,1)
            counter['calls']+=1
            capture=m.torch.cuda.is_current_stream_capturing()
            counter['captured_calls']+=int(capture)
            if context['row'] is not None:
                row=counter['full_layer_rows'].setdefault(context['row'],{'eager':0,'captured':0})
                row['captured' if capture else 'eager']+=1
            return call(data,bank,which,grid,mode)
        m.fx_call=counted
        run=m._best7_run
        def layer(data,state,plan,prefetch=None):
            c=data['config'];previous=context['row']
            if c['tokens']==256:
                assert plan['moe']=='w2fix_cta1g2112',plan
                context['row']=str((c['batch'],c['q_len'],c['window']))
            try:return run(data,state,plan,prefetch)
            finally:context['row']=previous
        m._best7_run=layer
        return m
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(candidate),'--no-baseline','--json',str(H/'forced_local.json')]
    try:local_eval.main()
    except SystemExit as e:
        if e.code==0:
            for key,plan in records['forced']['plans'].items():
                if key=='other':continue
                assert plan['moe']==('w2fix_cta1g2112' if key.startswith('(32, 8,') else 'v9nt1'),(key,plan)
            assert set(counter['full_layer_rows'])=={'(32, 8, 2048)','(32, 8, 0)'},counter
            assert all(row['captured']>0 and row['eager']>0 for row in counter['full_layer_rows'].values()),counter
            records.update(sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),w2fix_native_sm120=counter,
                           limit='Hopper attention/DG/Humming launches use established stand-ins; w2fix executes native portable math')
            (H/'forced_dispatch_local.json').write_text(json.dumps(records,indent=2)+'\n')
        raise
if __name__=='__main__':main()
