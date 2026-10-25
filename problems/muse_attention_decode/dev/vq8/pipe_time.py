"""Timed model of vq8's ring: producer (order 'new': publish it-LAG, wait empty(it-ST), issue it; order 'old':
wait empty, issue, publish it-LAG) vs consumer step rr (wait full(rr+1) while holding rr, compute C, release rr).
I = issue cycles, X = transpose cycles, M = load latency. Prints cycles per tile over a long portion."""
def period(ST, LAG, C, I, X, M, order='new', n=400):
    land, pub, rel = {}, {}, {}
    INF = float('inf')
    # Producer and consumer interleave through events; iterate to a fixed point (monotone times).
    for _ in range(4 * n):
        t = 0.0
        for it in range(n + LAG):
            def do_publish(t):
                tr = it - LAG
                if 0 <= tr < n:
                    t = max(t, land.get(tr, INF)) + X
                    pub[tr] = t
                return t
            def do_issue(t):
                if it < n:
                    if it >= ST:
                        t = max(t, rel.get(it - ST, INF))
                    t += I
                    land[it] = t + M
                return t
            if order == 'new':
                t = do_issue(do_publish(t))
            else:
                t = do_publish(do_issue(t))
            if t == INF:
                break
        c = max(0.0, pub.get(0, INF)) + C
        for rr in range(n - 1):
            c = max(c, pub.get(rr + 1, INF)) + C
            rel[rr] = c
        rel[n - 1] = c + C
        if all(k in rel and rel[k] < INF for k in range(n)):
            done = True
        else:
            done = False
        if done and _ > 3 and all(k in pub for k in range(n)):
            break
    a, b = n // 4, 3 * n // 4
    return (rel[b] - rel[a]) / (b - a)

if __name__ == '__main__':
    C, I, X, M = 1300, 500, 900, 4500
    print('C=%d I=%d X=%d M=%d (cycles)' % (C, I, X, M))
    for order in ('old', 'new'):
        for ST in (4, 6, 8):
            row = []
            for LAG in range(1, ST):
                try:
                    row.append('LAG%d=%.0f' % (LAG, period(ST, LAG, C, I, X, M, order)))
                except Exception as e:
                    row.append('LAG%d=ERR' % LAG)
            print(order, 'ST', ST, ' '.join(row))
