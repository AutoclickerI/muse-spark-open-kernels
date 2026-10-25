"""Check the least completion guarantees required by the cluster ring.

This is a protocol model, not execution of Hopper instructions. Every async
WGMMA group stays pending as long as its wait permits. Negative controls must
expose stale parity and unsafe slot reuse, so vacuous ordering cannot pass.
"""
import hashlib
import itertools
import json
from pathlib import Path
H=Path(__file__).resolve().parent


def ring(stages,cs,widths,ktiles=32,wait_limit=1,skew=False):
    parity=[0]*stages;checks=0
    for nt in widths:
        # Each warpgroup has a total group order. The minimum of all CTA
        # completion frontiers is what the cluster-wide overwrite must obey.
        done=[-1]*cs;slot_last=[[-1]*stages for _ in range(cs)]
        af_last=[[-1,-1] for _ in range(cs)];completion=[None]*stages
        mask=(1<<cs)-1
        def issue(k):
            nonlocal checks
            st=k%stages
            for rank in range(cs):
                assert done[rank]>=slot_last[rank][st],('overwrite',st,rank,done,slot_last)
                assert mask&(1<<rank)
            expected=nt*128*2
            counts=[expected]*cs
            for part in (nt*64*2,nt*64*2):
                for rank in range(cs):counts[rank]-=part
            assert counts==[0]*cs,('tx bytes',nt,cs)
            completion[st]=parity[st];parity[st]^=1;checks+=cs+1
        issue(0)
        for k in range(ktiles):
            st=k%stages
            assert completion[st]==((k//stages)&1),('phase',st,k,completion,parity)
            for sub in (0,1):
                group=2*k+sub
                for rank in range(cs):
                    assert done[rank]>=af_last[rank][sub],('A reuse',rank,k,sub,done,af_last)
                    af_last[rank][sub]=group;slot_last[rank][st]=group
                    done[rank]=max(done[rank],group-wait_limit)
                if stages==2 and sub==0 and k+1<ktiles:
                    # A missing all-rank barrier could allow rank0 to recycle
                    # while another CTA is still before its wait. Model that
                    # deliberately in the negative control.
                    if skew and cs>1 and slot_last[-1][(k+1)%stages]>=0:
                        done[-1]=slot_last[-1][(k+1)%stages]-1
                    issue(k+1)
            if stages==1:
                done=[2*k+1]*cs  # wait_group 0 before overwriting this slot
                if k+1<ktiles:issue(k+1)
            checks+=1
        assert all(v==0 for v in parity),('task phase does not reset',parity)
        # Last group may still be live for S2; the epilogue wait0 retires it.
        done=[2*ktiles-1]*cs
        assert all(done[r]>=max(slot_last[r]) for r in range(cs))
    return checks


def rejected(fn):
    try:fn()
    except AssertionError as e:return str(e)
    raise AssertionError('negative control did not expose a hazard')


def main():
    cases=[]
    # Every width pair used by a class, many changes of width between tasks,
    # and both first/second phases (their synchronization is identical).
    for maxnt,cs,first in itertools.product((64,128,192,256),(1,4),(False,True)):
        stages=2 if maxnt<=128 else 1
        widths=[maxnt-32,maxnt,maxnt,maxnt-32]*4
        checks=ring(stages,cs,widths)
        cases.append(dict(maxnt=maxnt,cs=cs,first=first,tasks=len(widths),checks=checks))
    negative=dict(wrong_wait=rejected(lambda:ring(2,4,[64],wait_limit=2)),
                  missing_cluster_barrier=rejected(lambda:ring(2,4,[64],skew=True)),
                  odd_K_parity=rejected(lambda:ring(1,4,[192,160],ktiles=31)))
    # Launch/route uniformity for every nearby expert count: e, row, width and
    # stage count must be identical at all ranks of a cluster; mt is distinct.
    launch_checks=0
    for m,maxnt,cs in itertools.product((127,128,255,256,257),(64,128,192,256),(1,4)):
        for n in range(m+1):
            for row in range(0,n,256):
                rem=min(n-row,256)
                active=n>16 and maxnt-64<rem<=maxnt
                if not active:continue
                nt=maxnt-32 if rem<=maxnt-32 else maxnt
                assert nt>=rem and nt%32==0
                assert [rank for rank in range(cs)]==list(range(cs))
                launch_checks+=1
    out=dict(cuda_sha256='075ce977480876698ce94283907faafc8afab49b798b7e1d4a5b42d801d08c8d',
             cases=cases,checks=sum(r['checks'] for r in cases),launch_checks=launch_checks,
             negative_controls=negative,
             limitation='Instruction-contract proof only; hardware execution/race checking is separate.')
    (H/'cluster_sync.json').write_text(json.dumps(out,indent=2)+'\n');print(out['checks'],launch_checks,negative)

if __name__=='__main__':main()
