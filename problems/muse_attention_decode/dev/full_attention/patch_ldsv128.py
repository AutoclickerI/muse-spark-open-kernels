"""Prepare an unpromoted S13 integration candidate while the paired H200 lab runs."""
import ast
import difflib
import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
p=R/'problems/muse_decode_layer/cand_r3_best7_s13_final.py';old=p.read_text()
assert hashlib.sha256(old.encode()).hexdigest().startswith('958f1258')
tree=ast.parse(old);lines=old.splitlines(keepends=True);edits=[]
cu=(H/'ldsv128pair.cu').read_text()
assert hashlib.sha256(cu.encode()).hexdigest().startswith('84eddcfe')
driver='''def _vq8_attention(qkv, data, setting):
    """Full verify16/32: adjacent one-head CTAs, native N128 raw-V register-A PV."""
    partial, splits = qkv
    cfg=data['config'];batch,length=cfg['batch'],cfg['q_len'];device=partial.device
    assert length==8 and not cfg['window'] and batch in (16,32)
    stages,min_tiles=setting
    if partial.dtype==torch.bfloat16:
        assert splits==1
        raw=partial[0] if partial.dim()==3 else partial
    else:
        assert partial.dtype==torch.float32 and partial.shape[0]==splits and partial.stride(-1)==1
        raw=partial.sum(0).to(torch.bfloat16)
    grid=3*_num_sms(device);segments=batch*(HKV//2)
    assert grid%2==0
    part=torch.empty(((grid//2+segments)*128*68+256,),device=device,dtype=torch.float32)
    out=torch.empty((batch*length,Q_SIZE),device=device,dtype=torch.bfloat16)
    _vq8_module().vq8_forward(raw.data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),
        data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),0,part.data_ptr(),out.data_ptr(),
        batch,length,HQ,HKV,raw.stride(-2),data['page_table'].shape[1],cfg['page_size'],0,0,
        min_tiles,grid,stages,cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,0,1,
        torch.cuda.current_stream(device).cuda_stream)
    return out
'''
for n in tree.body:
 name=getattr(n,'name',None)
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='VQ8_CUDA':
  edits.append((n.lineno-1,n.end_lineno,'VQ8_CUDA = r"""'+cu+'"""\n'))
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='_VQ8_ROWS':
  edits.append((n.lineno-1,n.end_lineno,'_VQ8_ROWS = {(16, 8, False): (3, 1), (32, 8, False): (3, 1)}\n'))
 if name=='_vq8_attention':edits.append((n.lineno-1,n.end_lineno,driver))
 if name=='_vq8_module':
  text=''.join(lines[n.lineno-1:n.end_lineno]).replace('muse_vq8_c262e4c5f9c3_','muse_vq8_84eddcfe7484_')
  edits.append((n.lineno-1,n.end_lineno,text))
 if name=='_vq8_prepare':
  text=''.join(lines[n.lineno-1:n.end_lineno]).replace("        print('VQ8_REJECT', str(exc).splitlines()[0][:200], flush=True)\n        return {}",'        raise RuntimeError("N128 register-V prepare gate failed") from exc')
  edits.append((n.lineno-1,n.end_lineno,text))
assert len(edits)==5
for a,b,s in sorted(edits,reverse=True):lines[a:b]=[s]
new=''.join(lines);compile(new,'s13_ldsv128.py','exec')
(H/'s13_ldsv128.py').write_text(new)
diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(p.relative_to(R)),tofile='b/'+str(p.relative_to(R))))
(H/'s13_ldsv128_experimental.patch').write_text(diff)
print('Unpromoted candidate',hashlib.sha256(new.encode()).hexdigest(),'patch',hashlib.sha256(diff.encode()).hexdigest())
