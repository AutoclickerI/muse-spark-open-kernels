"""Static S21 contracts and one-piece leave-outs before GPU work."""
import json,sys
from pathlib import Path
import compose
H=compose.H;c=compose.load(H.parent/'s19/contracts.py','s21_s19_contracts');c.compose=compose.s19
c.DELTA_TABLES.update({'DA_SELECTED','DA_NORMALIZED','DA_ROLES','DA_REQUESTS','_S20_META64',
                     '_P5B_FA_PICKS','_P5B_MERGE_SELECTION','_P5B_VQ_SELECTION',
                     '_P5B_FA_META','_P5B_FA_STORAGE','_P5B_FA_SMS','_PUBLIC_SELECTION'})
def units():return dict(S21=compose.build()[0],INC=compose.base(),**{'no_'+p:compose.build([x for x in compose.DEFAULT if x!=p])[0] for p in compose.DEFAULT})
def run(texts):
    base=c.Src(compose.base());old=c.Src(compose.expand(compose.P/'cand_r3_best7_s18_final.py'));host=c.Src(texts['S21']);checks=[]
    for name,text in texts.items():
        s=c.Src(text);checks += [dict(x,unit=name) for x in c.check_flags(s)+c.check_chain(s,old)]
        if name!='S21':checks+=c.check_plans(host,s,name)+c.check_banks(host,s,name)
        checks.append(dict(check='prepare_equal_S19',unit=name,ok=s.segs('prepare')==base.segs('prepare')))
        for key in ('P3B_CUDA','GLUE_CL_CUDA','RVC_NFIDX0_CUDA','RVC_NBASE2_CUDA'):
            checks.append(dict(check='dropped_piece_exact_S19',unit=name,key=key,ok=s.literal(key)==base.literal(key)))
        for fn in ('_boundary_fast','_boundary_late','smallm_experts','_best7_boundary'):
            checks.append(dict(check='dropped_dispatch_exact_S19',unit=name,fn=fn,ok=s.segs(fn)==base.segs(fn)))
        if '_p5b_sched_init' in s.defs:
            # These literal assignments share a semicolon line with the
            # deliberately different picks. Compare their values and producer.
            for key in ('_P5B_FA_META','_P5B_FA_STORAGE','_P5B_FA_SMS'):
                checks.append(dict(check='metadata_initial_value',unit=name,key=key,ok=s.literal(key)==host.literal(key)))
            checks.append(dict(check='metadata_producer',unit=name,ok=s.segs('_p5b_sched_init')==host.segs('_p5b_sched_init')))
    checks += [dict(x,unit='S21') for x in c.check_receipt(host)]
    checks += [dict(check='C3_exact',ok=compose.sha(host.literal('ATTN_PORT_CUDA_SRC'))=='519ea55f1acf936ac115e6e42b5d0c7abc6b913418865ebd2fe54d2d49932208'),
               dict(check='gate_rows',ok=host.literal('DA_SELECTED')=={(1,'gate_apply'):3}),
               dict(check='gate_sliding_guard',ok="state.get('da_window',False)" in host.segs('_gate_split_apply')[-1]),
               dict(check='superseded_FA_rows_absent',ok=host.literal('_P5B_FA_PICKS')=={} and host.literal('_P5B_MERGE_SELECTION')=={} and '_s20_m64_attention' not in host.defs),
               dict(check='VQ_rows',ok=host.literal('_P5B_VQ_SELECTION')=={(32,8,False):'Rw4'}),
               dict(check='public_rows',ok=host.literal('_PUBLIC_SELECTION')==compose.PUBLIC_SELECTION),
               dict(check='public_setup_in_pool',ok='_S19_BUILDS.append(_s21_public_build)' in texts['S21']),
               dict(check='unused_FA_bridge_not_built',ok='if _P5B_MERGE_SELECTION:' in host.segs('_p5b_merge_setup')[-1]),
               dict(check='NP_rows',ok=host.literal('NP_CONFIG')=={'qkv_gate':(128,64,64,8,4,1,3,False),'pool_router':(128,128,64,5,8,1,3,True)}),
               dict(check='NP_T128_guard',ok='x.shape[0]==128' in host.segs('_best7_project')[-1]),
               dict(check='incumbent_identity',ok=compose.sha(texts['INC'])==compose.BASE_SHA)]
    lane=c.Src((H/'inputs/chain.py.inc').read_text())
    for name in ('_P5B_BRIDGE_SRC','_P5B_REDUCE_SRC','_P5B_VQ_SOURCES'):
        checks.append(dict(check='measured_chain_source',name=name,ok=host.literal(name)==lane.literal(name)))
    ports=c.Src((H/'inputs/public_ports.py.inc').read_text())
    for name in ('_PUBLIC_SOURCES','_P5B_PUBLIC_NOTICES'):
        checks.append(dict(check='measured_licensed_port_source',name=name,ok=host.literal(name)==ports.literal(name)))
    checks.append(dict(check='license_notice_hash',ok=compose.sha(host.literal('_P5B_PUBLIC_NOTICES'))=='c91b7a513ad45d811b5658a368060cf32256271545d93c4cb0350404dafe1b4e'))
    for fn in ('_best7_attention','_public_module'):
        checks.append(dict(check='measured_port_glue',fn=fn,ok=host.segs(fn)[-1]==ports.segs(fn)[-1]))
    noport=c.Src(texts['no_PUBLIC'])
    checks.append(dict(check='public_leaveout_restores_prior_S21',ok='_PUBLIC_SELECTION' not in noport.tables and
                       noport.literal('_P5B_MERGE_SELECTION')=={(16,8,True):'R4w4',(1,8,False):'R64'} and
                       noport.literal('_P5B_VQ_SELECTION')=={(16,8,False):'Rw4',(32,8,False):'Rw4'}))
    for p in compose.DEFAULT:
        checks.append(dict(check='leaveout_identity',piece=p,ok=texts['no_'+p]!=texts['S21'] and texts['no_'+p]==compose.build([x for x in compose.DEFAULT if x!=p])[0]))
    return dict(passed=all(x['ok'] for x in checks),source_sha256=compose.sha(texts['S21']),checks=checks)
if __name__=='__main__':
    r=run(units());(H/'contracts.json').write_text(json.dumps(r,indent=1)+'\n');bad=[x for x in r['checks'] if not x['ok']]
    print('PASS' if r['passed'] else 'FAIL',len(r['checks']),'checks',bad[:15]);sys.exit(not r['passed'])
