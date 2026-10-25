"""Review-shaped cases from TASK B×L, public context/spread/page parameters."""
SEEDS = (419731, 427650)
S15_SHA = '602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'


def case(label, b, q=1, layer='sliding', ctx=80000, **kw):
    return dict(label=label, batch=b, q_len=q, layer=layer, ctx=ctx,
                spread=kw.pop('spread', 'agentic' if b>1 else 'fixed'),
                dist=kw.pop('dist','soft'), page=kw.pop('page',1), **kw)


def cases():
    out=[]
    for q in (1,8):
        for b in (1,16,32):
            for layer in ('sliding','full'):
                out.append(case(f'scored-{"d" if q==1 else "v"}{b}-{layer[0]}',b,q,layer))
    for q,bs in ((1,(2,8,24,48)),(8,(2,24))):
        for b in bs:
            for layer in ('sliding','full'):
                out.append(case(f'near-{"d" if q==1 else "v"}{b}-{layer[0]}',b,q,layer))
    # Just either side of table rows and token/CTA thresholds. Non-scored token
    # counts deliberately include odd rows; verify33 also exceeds the 256 gate.
    for b,q in ((7,1),(9,1),(15,1),(17,1),(31,1),(33,1),(63,1),(64,1),(65,1),(15,8),(17,8),(31,8),(33,8)):
        out.append(case(f'edge-{"d" if q==1 else "v"}{b}-s',b,q,ctx=4097))
    # Keep the selected kernels, change their spans. N128 and last-CTA dispatch
    # must not silently assume ctx=80000, page1, or a full window.
    for b,q in ((1,1),(16,1),(32,1),(16,8),(32,8)):
        for layer in ('sliding','full'):
            out.append(case(f'ctx64k-{b}x{q}-{layer[0]}',b,q,layer,65536))
    for ctx in (63,64,65,2040,2047,2048,2049):
        out.append(case(f'window-d1-{ctx}',1,ctx=ctx))
    for ctx in (2040,2047,2048,2049):
        out.append(case(f'window-v2-{ctx}',2,8,ctx=ctx,spread='fixed'))
    out += [case('page64-d1',1,ctx=80001,page=64),
            case('page64-v16-f',16,8,'full',ctx=8193,page=64),
            case('page64-v32-f',32,8,'full',ctx=4097,page=64,kv_scale=1/64),
            case('mixed-d24-f',24,layer='full',ctx=20001,spread='mixed'),
            case('mixed-v24-s',24,8,ctx=5001,spread='mixed'),
            case('ctx96k-v24-f',24,8,'full',96000),
            case('ctx96k-d48-f',48,1,'full',96000)]
    assert len({c['label'] for c in out})==len(out)
    return out
