"""Static S21 contracts and one-piece leave-outs before GPU work."""
import json,sys
from pathlib import Path
import compose
H=compose.H;c=compose.load(H.parent/'s19/contracts.py','s21_s19_contracts');c.compose=compose.s19
c.DELTA_TABLES.update({'DA_SELECTED','DA_NORMALIZED','DA_ROLES','DA_REQUESTS','_S20_META64'})
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
    checks += [dict(x,unit='S21') for x in c.check_receipt(host)]
    checks += [dict(check='C3_exact',ok=compose.sha(host.literal('ATTN_PORT_CUDA_SRC'))=='519ea55f1acf936ac115e6e42b5d0c7abc6b913418865ebd2fe54d2d49932208'),
               dict(check='gate_rows',ok=host.literal('DA_SELECTED')=={(1,'gate_apply'):3}),
               dict(check='gate_sliding_guard',ok="state.get('da_window',False)" in host.segs('_gate_split_apply')[-1]),
               dict(check='M64_shape_guard',ok='(1,8,0)' in host.segs('_best7_attention')[-1]),
               dict(check='incumbent_identity',ok=compose.sha(texts['INC'])==compose.BASE_SHA)]
    for p in compose.DEFAULT:
        checks.append(dict(check='leaveout_identity',piece=p,ok=texts['no_'+p]!=texts['S21'] and texts['no_'+p]==compose.build([x for x in compose.DEFAULT if x!=p])[0]))
    return dict(passed=all(x['ok'] for x in checks),source_sha256=compose.sha(texts['S21']),checks=checks)
if __name__=='__main__':
    r=run(units());(H/'contracts.json').write_text(json.dumps(r,indent=1)+'\n');bad=[x for x in r['checks'] if not x['ok']]
    print('PASS' if r['passed'] else 'FAIL',len(r['checks']),'checks',bad[:15]);sys.exit(not r['passed'])
