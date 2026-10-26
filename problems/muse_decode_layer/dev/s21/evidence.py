"""Losslessly deduplicate equal quantized sample vectors; never omit a seed or arm."""
def s21_pack_samples(loads,names,scale):
    vectors=[];seen={};aliases={}
    for label,cell in loads.items():
        aliases[label]={}
        for name in names:
            values=tuple(round(row[name]*scale)-scale for row in cell['samples'])
            if values not in seen:
                seen[values]=len(vectors);vectors.append(list(values))
            aliases[label][name]=seen[values]
    return dict(sample_encoding='deduplicated-i32-vectors-v1',sample_vectors=vectors,sample_aliases=aliases)

def s21_decode_samples(raw):
    assert raw['sample_encoding']=='deduplicated-i32-vectors-v1'
    vectors=raw.pop('sample_vectors');aliases=raw.pop('sample_aliases');names=raw['sample_arms']
    assert set(aliases)==set(raw['sample_loads'])
    for label in raw['sample_loads']:
        n=raw['loads'][label]['n'];indices=aliases[label];assert set(indices)==set(names)
        assert all(isinstance(i,int) and 0<=i<len(vectors) and len(vectors[i])==n for i in indices.values())
        raw['loads'][label]['samples']=[{name:1+vectors[indices[name]][i]/raw['sample_scale'] for name in names} for i in range(n)]
    return raw


def s21_pack_evidence(raw):
    """Lossless columns for the existing six-decimal archive; never alter G0 samples.

    GPU event milliseconds are float32. Store integer nanoseconds plus any exact
    decimal residual, so conversion is reversible even for synthetic non-events.
    Split numeric columns into byte planes before XZ; keep the JSON structure.
    """
    import base64,functools,json,lzma,math,struct
    def event_us(ns):return round(struct.unpack('<f',struct.pack('<f',ns/1e6))[0]*1000,6)
    def diagnostics(x):
        if isinstance(x,list):return [diagnostics(v) for v in x]
        if not isinstance(x,dict):return x
        out={}
        for key,value in x.items():
            if key=='round_us':
                ns={n:[round(v*1000) for v in a] for n,a in value.items()}
                unit=functools.reduce(math.gcd,(v for a in ns.values() for v in a),0) or 1
                corrections=[[n,i,round(v*1e6)-round(event_us(ns[n][i])*1e6)]
                             for n,a in value.items() for i,v in enumerate(a) if event_us(ns[n][i])!=v]
                out['round_cuda_samples']=dict(unit_ns=unit,ticks={n:[v//unit for v in a] for n,a in ns.items()},corrections_ps=corrections)
            elif key=='host_enqueue_us':
                ps={n:[round(v*1e6) for v in a] for n,a in value.items()}
                unit=functools.reduce(math.gcd,(v for a in ps.values() for v in a),0) or 1
                out['host_enqueue_samples']=dict(unit_ps=unit,ticks={n:[v//unit for v in a] for n,a in ps.items()})
            else:out[key]=diagnostics(value)
        return out
    integers=[];decimals=[]
    def split(x):
        if type(x) is int:integers.append(x);return 0
        if type(x) is float:
            assert math.isfinite(x) and x==round(x,6),x
            decimals.append(round(x*1e6));return 1
        if isinstance(x,list):return [split(v) for v in x]
        if isinstance(x,dict):
            assert all(isinstance(k,str) for k in x)
            return {k:split(v) for k,v in x.items()}
        assert x is None or isinstance(x,(str,bool)),type(x)
        return x
    shape=json.dumps(split(diagnostics(raw)),separators=(',',':')).encode()
    def planes(values):
        data=struct.pack('<'+'q'*len(values),*values)
        return b''.join(data[i::8] for i in range(8))
    data=struct.pack('<III',len(shape),len(integers),len(decimals))+shape+planes(integers)+planes(decimals)
    return dict(evidence_codec='s21-columns-v1',blob=base64.b85encode(lzma.compress(data,preset=9)).decode(),
                complete=raw['complete'],failures=list(raw['failures']))


def s21_unpack_evidence(envelope):
    import base64,json,lzma,struct
    assert envelope['evidence_codec']=='s21-columns-v1'
    data=lzma.decompress(base64.b85decode(envelope['blob']));size,ni,nf=struct.unpack('<III',data[:12])
    assert len(data)==12+size+8*(ni+nf)
    shape=json.loads(data[12:12+size]);offset=12+size
    def column(count):
        nonlocal offset
        buf=bytearray(8*count);packed=data[offset:offset+8*count];offset+=8*count
        for i in range(8):buf[i::8]=packed[i*count:(i+1)*count]
        return iter(struct.unpack('<'+'q'*count,buf))
    integers=column(ni);decimals=column(nf)
    def merge(x):
        if type(x) is int:
            assert x in (0,1)
            return next(integers) if x==0 else round(next(decimals)/1e6,6)
        if isinstance(x,list):return [merge(v) for v in x]
        if isinstance(x,dict):return {k:merge(v) for k,v in x.items()}
        return x
    raw=merge(shape);assert list(integers)==[] and list(decimals)==[]
    def diagnostics(x):
        if isinstance(x,list):return [diagnostics(v) for v in x]
        if not isinstance(x,dict):return x
        out={}
        for key,value in x.items():
            if key=='round_cuda_samples':
                unit=value['unit_ns'];rounds={n:[round(struct.unpack('<f',struct.pack('<f',v*unit/1e6))[0]*1000,6) for v in a] for n,a in value['ticks'].items()}
                for n,i,delta in value['corrections_ps']:rounds[n][i]=round((round(rounds[n][i]*1e6)+delta)/1e6,6)
                out['round_us']=rounds
            elif key=='host_enqueue_samples':out['host_enqueue_us']={n:[round(v*value['unit_ps']/1e6,6) for v in a] for n,a in value['ticks'].items()}
            else:out[key]=diagnostics(value)
        return out
    raw=diagnostics(raw)
    # paired_evidence attaches post-measurement device faults to this envelope.
    # Preserve them when exposing the archive to promotion.py.
    if not envelope['complete']:raw['complete']=False
    for fault in envelope['failures']:
        if fault not in raw['failures']:raw['failures'].append(fault)
    return raw
