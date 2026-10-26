def da_encode(archive):
    import json,struct,lzma,base64
    meta=dict(archive);rows=[];timing=bytearray();metrics=[]
    def put(value):
        z=value*2 if value>=0 else -value*2-1
        while z>=128:timing.append((z&127)|128);z>>=7
        timing.append(z)
    for row in archive['records']:
        r=dict(row);values=r.pop('values')
        for values_for_arm in values:
            first=values_for_arm[0];put(first)
            for value in values_for_arm[1:]:put(value-first)
        for field in ('errors','probes'):
            if field in r:
                values=r.pop(field);r[field+'_offset']=len(metrics);r[field+'_count']=len(values);metrics.extend(values)
        rows.append(r)
    meta.update(records=rows,timing_bytes=len(timing),metric_count=len(metrics))
    header=json.dumps(meta,separators=(',',':')).encode()
    blob=b'DA2\0'+struct.pack('<I',len(header))+header+timing+struct.pack('<'+str(len(metrics))+'f',*metrics)
    return base64.b85encode(lzma.compress(blob,preset=9)).decode()

def da_decode(encoded):
    import json,struct,lzma,base64
    blob=lzma.decompress(base64.b85decode(encoded));assert blob[:4]==b'DA2\0'
    n=struct.unpack_from('<I',blob,4)[0];meta=json.loads(blob[8:8+n]);start=8+n;position=start
    def get():
        nonlocal position
        z=shift=0
        while True:
            b=blob[position];position+=1;z|=(b&127)<<shift
            if b<128:break
            shift+=7
        return z//2 if z%2==0 else -(z//2)-1
    metrics=struct.unpack_from('<'+str(meta['metric_count'])+'f',blob,start+meta['timing_bytes'])
    for row in meta['records']:
        values=[]
        for arm in row['arms']:
            first=get();values.append([first]+[first+get() for _ in meta['seeds'][1:]])
        row['values']=values
        for field in ('errors','probes'):
            if field+'_offset' in row:
                offset=row.pop(field+'_offset');count=row.pop(field+'_count');row[field]=list(metrics[offset:offset+count])
    assert position==start+meta['timing_bytes'];return meta
