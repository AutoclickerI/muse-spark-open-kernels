"""Lossless integer-vector bit packing; keeps every round and telemetry read."""
import base64,math

def pack_vectors(vectors):
    n=len(vectors[0]);headers=[];blob=bytearray()
    for values in vectors:
        assert len(values)==n and all(isinstance(v,int) for v in values)
        base=min(values);step=math.gcd(*(v-base for v in values)) or 1
        offsets=[(v-base)//step for v in values];bits=max(offsets).bit_length();size=(n*bits+7)//8
        word=sum(v<<(i*bits) for i,v in enumerate(offsets));blob.extend(word.to_bytes(size,'little'));headers.append([base,step,bits])
    return dict(n=n,headers=headers,data85=base64.b85encode(blob).decode())

def unpack_vectors(packed):
    blob=base64.b85decode(packed['data85']);n=packed['n'];out=[];at=0
    for base,step,bits in packed['headers']:
        size=(n*bits+7)//8;word=int.from_bytes(blob[at:at+size],'little');at+=size;mask=(1<<bits)-1
        out.append([base+step*((word>>(i*bits))&mask) for i in range(n)])
    assert at==len(blob);return out

def pack_samples(packet):
    times=[];telemetry=[]
    for row in packet['paired']:
        values=row.pop('round_ns');row['round_index']=len(times);times.extend(v for seed in values for v in seed)
        values=row.pop('telemetry');row['telemetry_index']=len(telemetry);telemetry.extend(list(v) for seed in values for v in zip(*seed))
    packet.update(sample_encoding='lossless-i32-gcd-bitvectors-v1',round_vectors=pack_vectors(times),telemetry_vectors=pack_vectors(telemetry))
    return packet

def unpack_samples(packet):
    assert packet['sample_encoding']=='lossless-i32-gcd-bitvectors-v1';times=unpack_vectors(packet['round_vectors']);tel=unpack_vectors(packet['telemetry_vectors']);n=len(packet['seed_ids'])
    for row in packet['paired']:
        i=row['round_index'];row['round_ns']=[times[i+3*j:i+3*j+3] for j in range(n)]
        i=row['telemetry_index'];row['telemetry']=[list(map(list,zip(*tel[i+5*j:i+5*j+5]))) for j in range(n)]
    return packet
