def combined_metadata(value):
    # Timing samples are integer event quanta and remain lossless. Only
    # diagnostic error/prepare/elapsed metadata is serialized to six digits.
    if isinstance(value,float):return float(format(value,'.6g'))
    if isinstance(value,list):return [combined_metadata(v) for v in value]
    if isinstance(value,dict):return {k:combined_metadata(v) for k,v in value.items()}
    return value

def combined_encode(data):
    import copy,struct
    header=copy.deepcopy(data);body=bytearray()
    def put(value):
        while value>=128:body.append((value&127)|128);value>>=7
        body.append(value)
    for name in ('oracle','candidate'):
        for row in header.get(name,{}).get('rows',[]):
            for key in ('producer','neighbour'):
                matrix=row.get(key)
                if matrix is None:continue
                lengths=[len(a) for a in matrix];row[key]={'lengths':lengths}
                for arm in matrix:
                    last=0
                    for value in arm:
                        delta=value-last;put(delta*2 if delta>=0 else -delta*2-1);last=value
    text=json.dumps(header,separators=(',',':')).encode();blob=b'LSC1'+struct.pack('<I',len(text))+text+body
    return base64.b85encode(lzma.compress(blob,preset=9)).decode()

def combined_decode(text):
    import struct
    blob=lzma.decompress(base64.b85decode(text));assert blob[:4]==b'LSC1';n=struct.unpack_from('<I',blob,4)[0];header=json.loads(blob[8:8+n]);offset=8+n
    def get():
        nonlocal offset
        value=shift=0
        while True:
            b=blob[offset];offset+=1;value|=(b&127)<<shift
            if b<128:break
            shift+=7
        return value//2 if value%2==0 else -(value//2)-1
    for name in ('oracle','candidate'):
        for row in header.get(name,{}).get('rows',[]):
            for key in ('producer','neighbour'):
                desc=row.get(key)
                if desc is None:continue
                matrix=[]
                for count in desc['lengths']:
                    last=0;arm=[]
                    for i in range(count):last+=get();arm.append(last)
                    matrix.append(arm)
                row[key]=matrix
    assert offset==len(blob),(offset,len(blob));return header
