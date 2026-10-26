"""Lossless bit packing of the existing quantized per-seed sample vectors."""
import base64
def p3a_orders(names,rounds,seed_index):
 out=[]
 for block in range(rounds//2):
  shift=(seed_index+block)%len(names);order=names[shift:]+names[:shift]
  if (seed_index+block)%2:order=order[::-1]
  out.extend([order,order[::-1]])
 return out
def p3a_pack_vectors(raw):
 out=dict(raw);vectors=out.pop('sample_vectors');mins=[];widths=[];lengths=[];chunks=[];refs=[]
 for i,values in enumerate(vectors):
  residual=values;ref=-1;low=min(values);w=(max(values)-low).bit_length()
  for j,prior in enumerate(vectors[:i]):
   if len(prior)!=len(values):continue
   diff=[a-b for a,b in zip(values,prior)];dl=min(diff);dw=(max(diff)-dl).bit_length()
   if dw<w:residual,ref,low,w=diff,j,dl,dw
  number=sum((v-low)<<(j*w) for j,v in enumerate(residual)) if w else 0
  mins.append(low);widths.append(w);lengths.append(len(values));refs.append(ref);chunks.append(number.to_bytes((w*len(values)+7)//8,'little'))
 out['p3a_sample_codec']='predictive-bitpack-v1';out['p3a_sample_min']=mins;out['p3a_sample_width']=widths;out['p3a_sample_length']=lengths;out['p3a_sample_ref']=refs;out['p3a_sample_bits']=base64.b85encode(b''.join(chunks)).decode()
 if 'timing_diagnostics' in raw:
  diag=dict(raw['timing_diagnostics']);diag['traces']=[]
  for old in raw['timing_diagnostics']['traces']:
   trace=dict(old);assert trace.pop('orders')==p3a_orders(raw['sample_arms'],raw['measurement_protocol']['rounds'],trace['seed_index']);trace['p3a_protocol_orders']=True;diag['traces'].append(trace)
  out['timing_diagnostics']=diag
 return out
def p3a_unpack_vectors(raw):
 out=dict(raw);assert out.pop('p3a_sample_codec')=='predictive-bitpack-v1';mins=out.pop('p3a_sample_min');widths=out.pop('p3a_sample_width');lengths=out.pop('p3a_sample_length');refs=out.pop('p3a_sample_ref');buf=base64.b85decode(out.pop('p3a_sample_bits'));at=0;vectors=[]
 assert len(mins)==len(widths)==len(lengths)==len(refs)
 for i,(low,w,n,ref) in enumerate(zip(mins,widths,lengths,refs)):
  assert 0<=w<=63 and 0<n<=256;size=(w*n+7)//8;assert at+size<=len(buf);number=int.from_bytes(buf[at:at+size],'little');at+=size;mask=(1<<w)-1
  assert -1<=ref<i
  vectors.append([low+((number>>(j*w))&mask)+(vectors[ref][j] if ref>=0 else 0) for j in range(n)])
 if 'timing_diagnostics' in out:
  diag=dict(out['timing_diagnostics']);diag['traces']=[]
  for old in out['timing_diagnostics']['traces']:
   trace=dict(old);assert trace.pop('p3a_protocol_orders');trace['orders']=p3a_orders(out['sample_arms'],out['measurement_protocol']['rounds'],trace['seed_index']);diag['traces'].append(trace)
  out['timing_diagnostics']=diag
 assert at==len(buf);out['sample_vectors']=vectors;return out
