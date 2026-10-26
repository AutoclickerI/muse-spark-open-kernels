"""Bounded seed-ratio receipt; independent A:A retains absolute event times."""
def review_pack_v1(record):
 import base64,json,lzma,struct
 full=[r for r in record['rows'] if len(r)==7];nodes=[r for r in record['rows'] if len(r)==5]
 meta={k:v for k,v in record.items() if k not in ('rows','AA_rows')}
 meta.update(binary_full_rows=len(full),binary_node_rows=len(nodes),binary_AA_rows=len(record['AA_rows']))
 text=json.dumps(meta,separators=(',',':')).encode();data=struct.pack('<I',len(text))+text
 for r in full:data+=struct.pack('<HIII',r[0]|((1,8,16,32).index(r[1])<<4)|(r[2]<<6)|((r[3]+1)<<9),*r[4:])
 for r in nodes:data+=struct.pack('<HI',r[0]|((1,8,16,32).index(r[1])<<4)|(r[2]<<6),r[4])
 for r in record['AA_rows']:data+=struct.pack('<HIII',r[0]|((1,8,16,32).index(r[1])<<4),*r[2:])
 return base64.b85encode(lzma.compress(data,preset=9)).decode()

def review_pack(record):
 import base64,json,struct
 # Raw base85 has an entropy-independent bound. H200 clock variation made the
 # old compressed absolute-time matrix exceed the wrapper's 9000-char cap.
 full=[r for r in record['rows'] if len(r)==7]
 meta={k:record[k] for k in ('base','seeds','requested_seeds','arm_names','gates','failures','build_seconds','elapsed_seconds')}
 meta.update(version=2,binary_full_rows=len(full),binary_AA_rows=len(record['AA_rows']),
             weights_seed='175353+7919*seed',route_seed='179000+M+97*seed',
             transport='Per-seed median ratio at1ppm; A:A event times at1ns; candidate absolute times and CUPTI nodes omitted')
 text=json.dumps(meta,separators=(',',':')).encode()
 assert len(text)<=1600 and len(full)<=640 and len(record['AA_rows'])<=64
 data=struct.pack('<I',len(text))+text
 for r in full:data+=struct.pack('<HI',r[0]|((1,8,16,32).index(r[1])<<4)|(r[2]<<6)|((r[3]+1)<<9),r[6])
 for r in record['AA_rows']:data+=struct.pack('<HIII',r[0]|((1,8,16,32).index(r[1])<<4),*r[2:])
 archive='R2:'+base64.b85encode(data).decode()
 assert len(archive)<=7928  # ceil((4+1600+640*6+64*14)*5/4)+3
 return archive

def review_unpack(archive):
 import base64,json,lzma,struct
 if archive.startswith('R2:'):
  data=base64.b85decode(archive[3:]);length=struct.unpack_from('<I',data)[0];meta=json.loads(data[4:4+length]);pos=4+length;rows=[];aa=[]
  for _ in range(meta['binary_full_rows']):
   key,ratio=struct.unpack_from('<HI',data,pos);pos+=6
   rows.append([key&15,(1,8,16,32)[(key>>4)&3],(key>>6)&7,((key>>9)&15)-1,None,None,ratio])
  for _ in range(meta['binary_AA_rows']):
   key,a,b,ratio=struct.unpack_from('<HIII',data,pos);pos+=14
   aa.append([key&15,(1,8,16,32)[(key>>4)&3],a,b,ratio])
  assert pos==len(data)
  meta.update(rows=rows,AA_rows=aa);return meta
 data=lzma.decompress(base64.b85decode(archive));length=struct.unpack_from('<I',data)[0];meta=json.loads(data[4:4+length]);pos=4+length;rows=[];aa=[]
 for _ in range(meta['binary_full_rows']):
  key,a,b,ratio=struct.unpack_from('<HIII',data,pos);pos+=14
  rows.append([key&15,(1,8,16,32)[(key>>4)&3],(key>>6)&7,((key>>9)&15)-1,a,b,ratio])
 for _ in range(meta['binary_node_rows']):
  key,value=struct.unpack_from('<HI',data,pos);pos+=6
  rows.append([key&15,(1,8,16,32)[(key>>4)&3],(key>>6)&7,-1,value])
 for _ in range(meta['binary_AA_rows']):
  key,a,b,ratio=struct.unpack_from('<HIII',data,pos);pos+=14
  aa.append([key&15,(1,8,16,32)[(key>>4)&3],a,b,ratio])
 assert pos==len(data)
 meta.update(rows=rows,AA_rows=aa);return meta
