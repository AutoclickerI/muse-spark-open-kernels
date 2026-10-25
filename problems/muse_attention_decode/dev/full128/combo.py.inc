class _P5LiveS11:
    def __init__(self,ns):self.ns=ns
    @property
    def __dict__(self):return self.ns
    def __getattr__(self,key):return self.ns[key]

def lab(deadline,weights,state):
    global TL_EXISTING_HOST
    started=time.monotonic();TL_EXISTING_HOST=(_P5LiveS11(kernel_fn.__globals__),state)
    lines=['META frozen S11 acda6c62 actual decode weights/state; full N128 fixed ownership and nonspinning last-CTA sliding; 16 seeds x8 rotating read-cold whole-layer pairs.',
        'META A2=noise, PP=already-handed-off exact direct-partials attention; original CUPTI completion increments; native gates before timing. One shared prepare, no second S11 tuner.']
    errors=[];sliding=[]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        lf=pool.submit(cs_module);ppf=pool.submit(partial_port_ext,False)
        try:p5_full_pair(deadline,weights,state)
        except Exception as exc:
            errors.append('FAIL full '+type(exc).__name__+': '+str(exc)[:240]);print(errors[-1],flush=True)
        lf.result();ppf.result()
    try:sliding=p5_sliding_pair(deadline,weights,state)
    except Exception as exc:
        errors.append('FAIL sliding '+type(exc).__name__+': '+str(exc)[:240]);print(errors[-1],flush=True)
    packed_sliding=[]
    for l in sliding:
        if l.startswith('CS_XZ85 '):packed_sliding=json.loads(lzma.decompress(base64.b85decode(l.split(' ',1)[1])))
        if l.startswith('L '):lines.append(l)
    for r in PF_RESULTS:
        lines.append('F '+r['case']+' A=%.2f '%r['A_us']+' '.join(n+' r=%.5f inc=%.2f'%(r['r'][n],r['inc'][n]) for n in ('T3','T4'))+' Ainc=%.2f noise=%.5f'%(r['inc']['A'],r['r']['A2']))
    def rounded(x):
        if isinstance(x,float):return round(x,5)
        if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [rounded(v) for v in x]
        return x
    archive=dict(full=PF_RESULTS,sliding=packed_sliding,errors=errors,complete=len(PF_RESULTS)==2 and len(CS_RESULTS)==3 and not errors)
    blob=base64.b85encode(lzma.compress(json.dumps(rounded(archive),separators=(',',':')).encode(),preset=9)).decode()
    lines+=errors+['P5_XZ85 '+blob,'END full=%d/2 sliding=%d/3 elapsed=%.1fs'%(len(PF_RESULTS),len(CS_RESULTS),time.monotonic()-started)]
    assert sum(len(l)+1 for l in lines)<9000,('archive too large',len(blob))
    return lines
