def encode_rounds(result):
 # Exact signed deltas and unsigned LEB128. No rounding, filtering or samples
 # removed: all integer CUDA-event nanoseconds reconstruct byte-for-byte.
 rows=result.pop('round_ns');unit=0
 for samples in rows.values():
  for sample in samples:
   for values in sample.values():
    for v in values:unit=math.gcd(unit,v)
 unit=max(1,unit);order=list(rows);names={c:list(rows[c][0]) for c in order};buf=bytearray()
 for cell in order:
  for sample in rows[cell]:
   for name in names[cell]:
    prev=0
    for value in sample[name]:
     value//=unit;delta=value-prev;prev=value;n=2*delta if delta>=0 else -2*delta-1
     while n>=128:buf.append((n&127)|128);n>>=7
     buf.append(n)
 result['round_codec']=dict(version=1,unit_ns=unit,cells=order,names=names,seeds=len(result['seeds']),rounds={c:len(rows[c][0][names[c][0]]) for c in order},stream_xz85=base64.b85encode(lzma.compress(bytes(buf))).decode(),sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest())
 return result

def decode_rounds(result):
 if 'round_ns' in result:return result
 codec=result.pop('round_codec');assert codec['version']==1;buf=lzma.decompress(base64.b85decode(codec['stream_xz85']));at=0;rows={}
 for cell in codec['cells']:
  samples=[]
  for seed in range(codec['seeds']):
   sample={}
   for name in codec['names'][cell]:
    values=[];prev=0
    for i in range(codec['rounds'][cell]):
     n=0;shift=0
     while True:
      b=buf[at];at+=1;n|=(b&127)<<shift;shift+=7
      if not b&128:break
     prev+=(n>>1) if not n&1 else -(n>>1)-1;values.append(prev*codec["unit_ns"])
    sample[name]=values
   samples.append(sample)
  rows[cell]=samples
 assert at==len(buf)
 assert hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()==codec['sha256']
 result['round_ns']=rows;return result
