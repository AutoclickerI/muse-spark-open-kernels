# Lossless integer-column delta codec; every round and CUPTI nanosecond retained.
def nb_encode(value):
 columns={}
 def tree(x,path):
  if type(x) is int:columns.setdefault(path,[]).append(x);return 0
  if isinstance(x,dict):return {k:tree(v,path+'.'+k) for k,v in x.items()}
  if isinstance(x,list):return [tree(v,path+'[]') for v in x]
  return x
 skeleton=tree(value,'');header=json.dumps([skeleton,[[k,len(v)] for k,v in columns.items()]],separators=(',',':')).encode();data=bytearray(struct.pack('<I',len(header))+header)
 for values in columns.values():
  previous=0
  for v in values:
   delta=v-previous;previous=v;n=2*delta if delta>=0 else -2*delta-1
   while n>=128:data.append((n&127)|128);n>>=7
   data.append(n)
 return base64.b85encode(lzma.compress(data)).decode()
def nb_decode(blob):
 data=lzma.decompress(base64.b85decode(blob));size=struct.unpack_from('<I',data)[0];skeleton,layout=json.loads(data[4:4+size]);pos=4+size;columns={}
 for key,count in layout:
  values=[];previous=0
  for _ in range(count):
   n=shift=0
   while True:
    b=data[pos];pos+=1;n|=(b&127)<<shift
    if b<128:break
    shift+=7
   previous+=(n//2 if n%2==0 else -(n//2)-1);values.append(previous)
  columns[key]=iter(values)
 assert pos==len(data)
 def tree(x,path):
  if type(x) is int:return next(columns[path])
  if isinstance(x,dict):return {k:tree(v,path+'.'+k) for k,v in x.items()}
  if isinstance(x,list):return [tree(v,path+'[]') for v in x]
  return x
 return tree(skeleton,'')

# Event clocks have a common exact tick divisor (usually32ns). Factor it out
# after measurement; every round reconstructs bit for bit, without rounding.
def nb_time_encode(raw):
 wire=json.loads(json.dumps(raw));divisor=0
 for row in wire['rows']:
  for sample in row['times']:
   for values in sample.values():
    for v in values:divisor=math.gcd(divisor,v)
 assert divisor>0
 wire['wire_time_divisor']=divisor
 for row in wire['rows']:
  for sample in row['times']:
   for key,values in sample.items():sample[key]=[v//divisor for v in values]
 return nb_encode(wire)
def nb_time_decode(blob):
 raw=nb_decode(blob);divisor=raw.pop('wire_time_divisor')
 for row in raw['rows']:
  for sample in row['times']:
   for key,values in sample.items():sample[key]=[v*divisor for v in values]
 return raw
