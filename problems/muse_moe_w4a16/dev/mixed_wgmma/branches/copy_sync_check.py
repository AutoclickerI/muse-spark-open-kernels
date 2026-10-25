"""Worst allowed copy/WGMMA completion for S1/S2/S3 direct-copy rings."""
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent


def check(stages, ktiles=32, empty_tail=True, wait=1):
    copies = []
    ready = set()
    slot = [None] * stages
    readers = [-1] * stages
    af_readers = [-1, -1]
    done = -1
    checks = 0

    def issue(k):
        nonlocal checks
        st = k % stages
        assert done >= readers[st], ('overwrite live X', stages, k)
        slot[st] = k
        copies.append(k)
        checks += 1

    for p in range(max(stages - 1, 1)):
        issue(p)
    for k in range(ktiles):
        # wait_group N guarantees all except the N newest committed groups.
        pending = max(stages - 2, 0)
        while len(copies) > pending:
            ready.add(copies.pop(0))
        assert k in ready and slot[k % stages] == k, ('copy not complete', stages, k)
        for sub in (0, 1):
            assert done >= af_readers[sub], ('A buffer live', stages, k, sub)
            g = 2 * k + sub
            readers[k % stages] = g
            af_readers[sub] = g
            done = max(done, g - wait)
            if stages > 1 and sub == 0:
                future = k + stages - 1
                if future < ktiles:
                    issue(future)
                elif empty_tail:
                    copies.append(None)
            checks += 1
        if stages == 1:
            done = 2 * k + 1
            if k + 1 < ktiles:
                issue(k + 1)
    done = 2 * ktiles - 1
    assert done >= max(readers)
    return checks


def rejects(fn):
    try:
        fn()
    except AssertionError as e:
        return str(e)
    raise AssertionError('negative control did not fail')


def main():
    cases = [dict(stages=s, ktiles=k, checks=check(s, k))
             for s in (1, 2, 3) for k in range(3, 66)]
    out = dict(cases=cases, checks=sum(x['checks'] for x in cases),
               negative_controls=dict(
                   no_empty_tail=rejects(lambda: check(3, empty_tail=False)),
                   wrong_wgmma_wait=rejects(lambda: check(3, wait=2))),
               deep_cuda_sha256=hashlib.sha256((H/'cu_cluster_deep.cu').read_bytes()).hexdigest(),
               limitation='Instruction-contract model; local/H200 execution checked separately.')
    (H/'copy_sync.json').write_text(json.dumps(out, indent=2)+'\n')
    print(out['checks'], out['negative_controls'])


if __name__ == '__main__':
    main()
