"""S15L: selected library boundaries replaced by existing owned arithmetic."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent;P=H.parents[1]
BASE=P/'cand_r3_best7_s15.py'
SHA='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
GROUPS=('o','pool128','unpool128','v1full','v32slide')


def one(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)


def compose(without=()):
    s=BASE.read_text();assert hashlib.sha256(s.encode()).hexdigest()==SHA
    if 'o' not in without:
        s=one(s,"NF_PORT_FORCE = False", "NF_PORT_SELECTED |= {(128, 'o'), (256, 'o')}\nNF_PORT_FORCE = False")
    vrows={}
    if 'v1full' not in without:vrows[1,8,False]=(4,2)
    if 'v32slide' not in without:vrows[32,8,True]=(4,2)
    if vrows:
        old="_VQ8_ROWS = {(16, 8, False): (4, 8), (32, 8, False): (4, 8)}"
        s=one(s,old,old+'\n_VQ8_ROWS.update('+repr(vrows)+')')
    roles={}
    if 'pool128' not in without:roles[128,'pool_router']=('vdsk',(128,128,64,5,8,1),3)
    if 'unpool128' not in without:roles[128,'unpooler']=('vdsk',(128,128,64,5,8,0),2)
    # The existing qkv VDSK launch keeps CHAIN=False, with no new instruction.
    # Replaced dependent GEMMs release at entry and wait before input/TMA reads.
    if roles:
        s=one(s,'NS: tl.constexpr, TMA: tl.constexpr):\n    pid = tl.program_id(0)',
            'NS: tl.constexpr, TMA: tl.constexpr, CHAIN: tl.constexpr = False):\n    if CHAIN:\n        tl.extra.cuda.gdc_launch_dependents()\n        tl.extra.cuda.gdc_wait()\n    pid = tl.program_id(0)')
        s=one(s,'def _vdsk_project(x, w, variant):','def _vdsk_project(x, w, variant, chain=False):')
        s=one(s,'num_warps=NWARPS, num_stages=NS)\n    return P, S',
            'num_warps=NWARPS, num_stages=NS, CHAIN=chain, launch_pdl=chain)\n    return P, S')
        # Add at the final dispatch only: all other roles/plans are untouched.
        marker='def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):\n    key=(x.shape[0],name)\n'
        s=one(s,marker,'LIBRARY_NATIVE_ROWS = '+repr(roles)+'\n\n'+marker+
            '    if key in LIBRARY_NATIVE_ROWS:\n        return _vdsk_project(x,state[name],LIBRARY_NATIVE_ROWS[key],chain=True)\n')
    return s


def main():
    out=H/'out';out.mkdir(exist_ok=True);arms={}
    for name,without in [('L',())]+[('no_'+g,(g,)) for g in GROUPS]:
        s=compose(without);p=P/'cand_r3_best7_s15L.py' if name=='L' else out/(name+'.py');p.write_text(s)
        arms[name]=dict(sha256=hashlib.sha256(s.encode()).hexdigest(),without=without)
    diff=list(difflib.unified_diff(BASE.read_text().splitlines(),compose().splitlines()))
    meta=dict(base_sha256=SHA,arms=arms,groups=GROUPS,diff=dict(
        add=sum(l.startswith('+') and not l.startswith('+++') for l in diff),
        remove=sum(l.startswith('-') and not l.startswith('---') for l in diff)))
    # Every CUDA source and every existing dispatch table stays identical except
    # the two new attention rows and two NF O rows. No changed vendor arithmetic.
    def literals(src):
        return {n.targets[0].id:n.value.value for n in ast.parse(src).body if isinstance(n,ast.Assign)
            and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)
            and isinstance(n.value.value,str)}
    a,b=literals(BASE.read_text()),literals(compose());assert a==b
    meta['all_string_literals_identical']=len(a)
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)


if __name__=='__main__':main()
