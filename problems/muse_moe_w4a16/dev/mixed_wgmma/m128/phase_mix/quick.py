"""Exercise standalone NT1 phases and each mixed composition before screening."""
import importlib.util
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    m=load('pm_quick',ROOT/'prof/lab/moe_m128_phase_mix.py')
    w=m.dc_weights(5353,'cuda');w['w13_gs_up']=w['w13_gs'];w['w13_gs_gate']=w['w13_gs']
    base=m.issue_prepare(w);word=m.issue_word_bank(base);native=m.issue_nt_prepare(w);ds=m.dc_fixtures(w,128);want=m.dc_oracles(ds[:1],w)[0]
    for name,a,b in m.PM_CHOICES:
        plan=m.pm_plan(ds[0],base,word,native,a,b)
        for fn in plan['steps']:out=fn()
        torch.cuda.synchronize();e,r=m.pm_error(out,want);print(name,e,r,flush=True)
        assert e<=.01 and r<=.05,(name,e,r)
    cell=load('d32g_handoff',HERE/'d32g_cell.py');bank=cell.d32g_prepare(w)
    graph=m.MWGraph(lambda d:cell.d32g_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],bank),ds[0])
    out=graph.replay();torch.cuda.synchronize();e,r=m.pm_error(out,want);assert e<=.01 and r<=.05,(e,r)
    print('D32G_HANDOFF_PASS',e,r,flush=True)


if __name__=='__main__':main()
