"""Simulate vq8's producer (publish tile it-LAG, then wait slot, issue it) against the consumer's step order
(first tile: wait full[r]; step(rr): wait full[rr+1] ... release rr; finish: release last) over portions."""
import itertools

def run(ST, LAG, portions):
    nreal = sum(portions)
    # producer program: list of ops
    prod = []
    for it in range(nreal + LAG):
        tr = it - LAG
        if 0 <= tr < nreal:
            prod.append(('publish', tr))
        if it < nreal:
            if it >= ST:
                prod.append(('wait_empty', it - ST))
            prod.append(('issue', it))
    cons = []
    r = 0
    for n in portions:
        if n == 0:
            continue
        cons.append(('wait_full', r))
        for j in range(n - 1):
            cons.append(('wait_full', r + j + 1))
            cons.append(('release', r + j))
        cons.append(('release', r + n - 1))
        r += n
    published, released, issued = set(), set(), set()
    pi = ci = 0
    while pi < len(prod) or ci < len(cons):
        progressed = False
        while pi < len(prod):
            op, t = prod[pi]
            if op == 'wait_empty' and t not in released:
                break
            if op == 'publish':
                assert t in issued, ('publish before issue', t)
                published.add(t)
            if op == 'issue':
                # slot reuse safety: tile t-ST must be released before t is issued
                assert t < ST or (t - ST) in released, ('slot reuse', t)
                issued.add(t)
            pi += 1
            progressed = True
        while ci < len(cons):
            op, t = cons[ci]
            if op == 'wait_full' and t not in published:
                break
            if op == 'release':
                released.add(t)
            ci += 1
            progressed = True
        if not progressed:
            return False
    return True

bad = []
for ST in (2, 3, 4, 6, 8):
    for LAG in (ST - 2, ST - 1, ST):
        if LAG < 1:
            continue
        ok = all(run(ST, LAG, list(p)) for n in range(1, 5) for p in itertools.product(range(0, 13), repeat=n) if sum(p) > 0 and len(p) <= 3)
        print('ST', ST, 'LAG', LAG, 'ok' if ok else 'DEADLOCK/violation')
