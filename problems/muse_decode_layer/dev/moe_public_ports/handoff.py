"""Verified negative handoff: retain S19; never manufacture a winner patch."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
main=json.loads((H/'result.json').read_text());abi=json.loads((H/'abi_result.json').read_text())
assert all(x=='A' for x in main['selection'].values()) and all(x=='A' for x in abi['selection'].values())
# Point gains whose intervals span one are diagnostic only. A finish timing
# over an unchanged S19 kernel is a control, not a new mechanism.
assert not any(q['hi']<1 for m in abi['phases'].values() for n,q in m.items() if n in ('O132:w13','O132:w2','LF:finish'))
for p in ('candidate.patch','nominee.py.inc'):(H/p).unlink(missing_ok=True)
sources=('operator.py.inc','operator_abi.py.inc','abi_helpers.py.inc','lee_x.cu.inc','lee_f.cu.inc','lee_repack.py.inc','abi_finalize.cu.inc','NOTICES.txt')
result=dict(status='retain S19 at every M; no winner patch or G0 nomination',base_sha256=main['identities']['base'],selection=main['selection'],receipts=[main['receipt'],abi['receipt']],checks=dict(primary=main['checks'],matched_ABI=abi['checks']),whole=main['whole'],matched_abi_whole=abi['whole'],matched_abi_phases=abi['phases'],AA=dict(primary=main['AA'],matched_ABI=abi['AA']),M1_32={m:dict(whole=main['whole'][m],native_phases=main['phases'][m],S19='Fused W13+W2+finish; separate phase denominators do not exist') for m in ('1','8','16','32')},sources_sha256={p:hashlib.sha256((H/p).read_bytes()).hexdigest() for p in sources},mechanism_notes='MECHANISMS.md',whole_layer_G0='unmeasured; no candidate warranted by these negative component cohorts',P3b='Public-port recipe closed. Native O W2 / Lee finish apparent gains fail matched ABI; O W13 M128 point unresolved. The structural ceiling remains open; no measured speedup mechanism reopens a build above 1% six-load.',licenses='Complete MIT, Apache-2.0 and operatorstack NOTICE text retained in NOTICES.txt and embedded reviewed Python/payloads; modified Lee files identified; no upstream Lee NOTICE at this pin.',candidate_patch=None)
(H/'P4_HANDOFF.json').write_text(json.dumps(result,indent=2)+'\n');print('P4: S19 all six M; no winner patch; matched ABI negative evidence included')
