"""Exercise the corrected tuner plus all Hopper adapters through the existing stand-ins."""
from pathlib import Path
import argparse,sys,json,time
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev/best6')]
import local_eval
import force as best6_force


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');args=ap.parse_args()
    old=local_eval.load_module;records={}
    def load(name,path):
        m=old(name,path)
        if name=='submission':
            records['base']=best6_force.install(m,args.cold_native)
            original=m.prepare
            def prepare(weights):
                state=original(weights)
                rows=state['attention_public_rows']
                assert len(rows)==12,[(r['batch'],r['q_len'],r['window']) for r in rows]
                for row in rows:
                    assert len(row['seq_lens'])==3 and len(row['seeds'])==3,row
                    assert all(len(t['seed_us'])==3 for t in row['trials']),row
                    assert any(t['choice']==('layer',None) for t in row['trials']),row
                    if row['batch']>1:
                        assert all(max(x)>min(x) for x in row['seq_lens']),row
                        assert len({tuple(x) for x in row['seq_lens']})==3,row
                records['rows']=rows
                print('ATTN_PUBLIC_FORCED',len(rows),'shapes with 3 draws each and layer control',flush=True)
                return state
            m.prepare=prepare
        return m
    local_eval.load_module=load
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/cand_r3_best6.py'),'--no-baseline',
              '--json',str(ROOT/'problems/muse_decode_layer/dev/best6_attention_forced_local.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:
            assert all(v>0 for row in records['base']['cycle']['choices'].values() for v in row.values()),records['base']
            (HERE/'forced_choices_local.json').write_text(json.dumps(records,indent=2)+'\n')
        raise
if __name__=='__main__':main()
