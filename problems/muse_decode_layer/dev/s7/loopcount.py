"""Hot K-loop SASS of H_H's shipped heavy kernels (idist fused NT32, W13 and W2), S6 vs S7."""
import re, collections, sys
def body(path, name):
    lines = []; cur = None
    for line in open(path):
        m = re.match(r'\s+Function : (\S+)', line)
        if m: cur = m.group(1); continue
        if cur == name:
            m = re.match(r'\s+/\*([0-9a-f]{4})\*/\s+(.*?)\s*;', line)
            if m: lines.append((int(m.group(1), 16), m.group(2).strip()))
    return lines
def op(t):
    t = re.sub(r'^@!?U?P[T0-9]+\s+', '', t); return t.split()[0]
names = [l.split(':')[1].strip() for l in open('hh_s6.sass') if 'Function' in l and 'idist5fused' in l and 'Li32E' in l and 'Li1EEEv' in l]
for name in names:
    for tag in ('s6', 's7'):
        b = body(f'hh_{tag}.sass', name)
        addr = {a: i for i, (a, t) in enumerate(b)}
        hg = [i for i, (a, t) in enumerate(b) if op(t).startswith('HGMMA') or op(t).startswith('WARPGROUP')]
        loops = []
        for i, (a, t) in enumerate(b):
            if op(t) == 'BRA':
                m = re.search(r'0x([0-9a-f]+)', t)
                if m and int(m.group(1), 16) < a and int(m.group(1), 16) in addr:
                    loops.append((addr[int(m.group(1), 16)], i))
        cand = [l for l in loops if sum(l[0] <= h <= l[1] for h in hg) >= 8]
        s, e = min(cand, key=lambda l: l[1] - l[0])
        ops = collections.Counter(op(t) for a, t in b[s:e + 1])
        n = sum(ops.values()); h = sum(v for k, v in ops.items() if k.startswith('HGMMA'))
        keys = ('PRMT', 'LEA', 'IMAD', 'IMAD.SHL.U32', 'SHF.L.U32', 'LOP3.LUT', 'HMUL2.BF16_V2', 'LDS.64', 'LDS', 'LDGSTS.E.64', 'LDGSTS.E', 'LDGSTS.E.BYPASS.128', 'IADD3', 'VIADD')
        print(('W13' if 'Lb1E' in name else 'W2 '), tag, f'loop={n} insts, HGMMA={h}, per16w={n/16:.2f}', {k: ops[k] for k in keys if ops[k]})
