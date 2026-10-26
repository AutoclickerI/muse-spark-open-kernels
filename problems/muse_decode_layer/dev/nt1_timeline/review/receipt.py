"""Binary encoding of the complete seed-level matrix; times1ns and ratios1ppm."""
def review_pack(record):
 import base64,json,lzma,struct
 full=[r for r in record['rows'] if len(r)==7];nodes=[r for r in record['rows'] if len(r)==5]
 meta={k:v for k,v in record.items() if k not in ('rows','AA_rows')}
 meta.update(binary_full_rows=len(full),binary_node_rows=len(nodes),binary_AA_rows=len(record['AA_rows']))
 text=json.dumps(meta,separators=(',',':')).encode();data=struct.pack('<I',len(text))+text
 for r in full:data+=struct.pack('<HIII',r[0]|((1,8,16,32).index(r[1])<<4)|(r[2]<<6)|((r[3]+1)<<9),*r[4:])
 for r in nodes:data+=struct.pack('<HI',r[0]|((1,8,16,32).index(r[1])<<4)|(r[2]<<6),r[4])
 for r in record['AA_rows']:data+=struct.pack('<HIII',r[0]|((1,8,16,32).index(r[1])<<4),*r[2:])
 return base64.b85encode(lzma.compress(data,preset=9)).decode()

def review_unpack(archive):
 import base64,json,lzma,struct
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
