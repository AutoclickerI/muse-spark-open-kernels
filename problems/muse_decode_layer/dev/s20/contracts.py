"""S19's static contracts extended for the seven S20 recipes; no GPU imports."""
import ast,json,sys
from pathlib import Path
import compose
H=Path(__file__).resolve().parent
c=compose.load(H.parent/'s19/contracts.py','s20_old_contracts')
c.compose=compose.s19
c.DELTA_TABLES.update({'DA_SELECTED','DA_NORMALIZED','DA_ROLES','DA_REQUESTS','_S20_META64'})
def units():
    return dict(S20=compose.build()[0],INC=compose.base(),**{'no_'+p:compose.build([x for x in compose.DEFAULT if x!=p])[0] for p in compose.DEFAULT})
def run(texts):
    old=c.Src(compose.expand(compose.P/'cand_r3_best7_s18_final.py'));host=c.Src(texts['S20']);checks=[]
    for name,text in texts.items():
        s=c.Src(text)
        checks += [dict(x,unit=name) for x in c.check_flags(s)+c.check_chain(s,old)]
        if name!='S20':checks+=c.check_plans(host,s,name)+c.check_banks(host,s,name)
        checks.append(dict(check='prepare_equal_S19',unit=name,ok=s.segs('prepare')==c.Src(compose.base()).segs('prepare')))
    checks += [dict(x,unit='S20') for x in c.check_receipt(host)]
    expected={p:'no_'+p for p in compose.DEFAULT}
    checks.append(dict(check='all_leaveouts',ok=all(v in texts for v in expected.values())))
    checks.append(dict(check='incumbent_identity',ok=compose.sha(texts['INC'])==compose.BASE_SHA))
    checks.append(dict(check='C3_exact',ok=compose.sha(host.literal('ATTN_PORT_CUDA_SRC'))=='519ea55f1acf936ac115e6e42b5d0c7abc6b913418865ebd2fe54d2d49932208'))
    checks.append(dict(check='gate_rows',ok=host.literal('DA_SELECTED')=={(1,'gate_apply'):3}))
    checks.append(dict(check='gate_window_guard',ok="state.get('da_window',False)" in host.segs('_gate_split_apply')[-1]))
    checks.append(dict(check='M64_shape_guard',ok="(1,8,0)" in host.segs('_best7_attention')[-1]))
    base=c.Src(compose.base());ew=compose.load(compose.EW/'patch.py','s20_contract_ew')
    checks.append(dict(check='H2_exact',ok=host.literal('P3B_CUDA')==ew.ew_patch_heavy(base.literal('P3B_CUDA'),w2=True)))
    checks.append(dict(check='B1_exact',ok=host.literal('GLUE_CL_CUDA')==ew.ew_patch_glue(base.literal('GLUE_CL_CUDA'),('b1',))))
    checks.append(dict(check='B1_late_retains_S19_revert',ok=host.segs('_boundary_late')==[compose.b1_triton(base.segs('_boundary_late')[0],ew)]))
    for p in compose.DEFAULT:
        # Each arm must differ and must be exactly the reproducible one-piece leave-out.
        arm=texts['no_'+p];checks.append(dict(check='leaveout_identity',piece=p,ok=arm!=texts['S20'] and arm==compose.build([x for x in compose.DEFAULT if x!=p])[0]))
    return dict(passed=all(x['ok'] for x in checks),checks=checks,components={p:dict(rows=compose.ROWS[p],leaveout='no_'+p) for p in compose.DEFAULT})
if __name__=='__main__':
    result=run(units());(H/'contracts.json').write_text(json.dumps(result,indent=1)+'\n')
    bad=[x for x in result['checks'] if not x['ok']];print('PASS' if result['passed'] else 'FAIL',len(result['checks']),'checks');print(bad[:30] if bad else '')
    sys.exit(not result['passed'])
