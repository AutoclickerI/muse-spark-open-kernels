"""Exercise all three ready lab entry points with portable standard-stage depth."""
import importlib.util
import json
from pathlib import Path
import time


def main():
    results={}
    for name in ('moe_nt1_decode_ready','moe_nt1_occupancy_ready','moe_nt1_branches'):
        spec=importlib.util.spec_from_file_location(name,Path('prof/lab')/(name+'.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        native=m.nt1_module
        class PortableDepth:
            def __init__(self,mod):self.mod=mod
            def __getattr__(self,name):return getattr(self.mod,name)
            def smem_needed(self,nt,st,rows):return self.mod.smem_needed(nt,min(st,2),rows)
            def resident_blocks(self,nt,st,rows):return self.mod.resident_blocks(nt,min(st,2),rows)
            def run(self,p,rows,start,st,grid,stream):self.mod.run(p,rows,start,min(st,2),grid,stream)
        def portable(kind):
            mod=native(kind)
            return PortableDepth(mod) if kind!='split2' else mod
        m.nt1_module=portable
        started=time.monotonic();w=m.rg_generate_weights(5353);w['post_norm_w']=w['post_expert_norm']
        lines=m.lab(started+78,w)
        assert not any('REJECT' in s or 'BUDGET' in s for s in lines),lines
        agg=[s for s in lines if s.startswith('AGG')]
        assert len(agg)==4 and all('seeds=12 ' in s for s in agg),lines
        results[name]=dict(seconds=time.monotonic()-started,full_seed_aggregates=4)
        print('LOCAL_ENTRY',name,results[name],flush=True)
        for line in agg+[lines[-2],lines[-1]]:print(line,flush=True)
    Path('runs/screen/nt1_branches_entries.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
