"""Consume a landed local receipt, preserving failed bit gates and null results."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
parser=argparse.ArgumentParser();parser.add_argument('receipt');args=parser.parse_args()
path=Path(args.receipt)
if not path.exists():
    hits=list(ROOT.glob('runs/*/*_'+args.receipt+'.json'))
    if not hits:raise SystemExit('No local receipt for '+args.receipt)
    path=hits[0]
receipt=json.loads(path.read_text());tail=receipt['result'].get('log_tail','')
block=re.search(r'^LAB_BEGIN generic_memory_compression (\w+)\n(.*?)^LAB_END.*?$',tail,re.M|re.S)
if not block:raise SystemExit('No compression lab in '+str(path))
manifest=json.loads((HERE/'manifest.json').read_text())
if block[1]!=manifest['payload_sha256'][:16]:raise SystemExit('Receipt/payload mismatch')
text=block[2];match=re.search(r'^SUPPORT (\{.*\})$',text,re.M)
if not match:raise SystemExit('No driver support result:\n'+text)
support=ast.literal_eval(match[1]);controls=[];kernels=[]
for line in text.splitlines():
    if line.startswith('BW '):
        v=re.match(r'BW (\w+) N/C=([\d.]+)/([\d.]+)us GB/s=([\d.]+)/([\d.]+) gain=([\d.]+) aa=([\d.]+).* all=([\d.,]+)',line)
        if not v:raise SystemExit('Unparsed bandwidth row: '+line)
        controls.append(dict(data=v[1],normal_us=float(v[2]),compressed_us=float(v[3]),
            normal_gbs=float(v[4]),compressed_gbs=float(v[5]),gain=float(v[6]),aa_ratio=float(v[7]),
            all_config_gains=list(map(float,v[8].split(',')))))
    if line.startswith(('MOE ','DENSE ')):
        v=re.search(r'N/C=([\d.]+)/([\d.]+)us r=([\d.]+).*aa=([\d.]+) bit=(\d) aa_bad=(\d+)/24 ab_bad=(\d+)/36 s=([\d.,]+)',line)
        if not v:raise SystemExit('Unparsed kernel row: '+line)
        kernels.append(dict(name=line.split(' N/C=',1)[0],normal_us=float(v[1]),compressed_us=float(v[2]),
            ratio=float(v[3]),aa_ratio=float(v[4]),bit_identical=bool(int(v[5])),aa_bad=int(v[6]),ab_bad=int(v[7]),
            seed_ratios=list(map(float,v[8].split(','))),ordered_bit_exact='ordered_bit=1' in line,
            w2_bit_exact='Ybit=1' in line,raw_line=line))
complete=bool(re.search(r'^END controls=6 kernels=4 ',text,re.M)) and 'MC_ERROR' not in text
if support.get('granted')==1 and complete and (len(controls)!=6 or len(kernels)!=4):
    raise SystemExit('Missing rows in the landed receipt')
real_gains=[r for r in controls if r['data'] in ('raw_nvfp4','recoded_nvfp4','bf16') and
            r['gain']>1.02 and min(r['all_config_gains'])>1.015]
cells=[r for r in kernels if r['ratio']<.985 and max(r['seed_ratios'])<1. and r['bit_identical']]
report=dict(receipt=str(path),payload_sha256=manifest['payload_sha256'],support=support,
    complete=complete,controls=controls,kernels=kernels,real_weight_bw_gains=real_gains,
    bit_exact_kernel_cells=cells,raw_lab=text,dropin=None)
if complete and support.get('granted')==1 and (real_gains or cells):
    allocation=(HERE/'allocation.py.inc').read_text()
    if hashlib.sha256(allocation.encode()).hexdigest()!=manifest['allocation_sha256']:
        raise SystemExit('Allocator changed since measurement')
    body='"""Prepare-only compressed weight copies; retain allocation owner with state."""\n'+allocation+'\n'
    body+='MC_CUDA='+repr((HERE/'kernel.cu').read_text())+'\n'+(HERE/'runtime.py.inc').read_text()+'\n'
    body+='''
def compressed_weight_copy(weights):
    """Return (same-key tensor views, owner); allocate/copy only during prepare."""
    offsets={};size=0
    for name,tensor in weights.items():
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.nbytes%16:
            raise ValueError('expected contiguous CUDA weights with complete vectors')
        size=(size+255)//256*256;offsets[name]=size;size+=tensor.nbytes
    owner=MappedAllocation((size+15)//16*16,1)
    try:
        if owner.granted!=1:raise RuntimeError('driver did not grant generic compression')
        out={}
        for name,tensor in weights.items():
            out[name]=owner.tensor(tuple(tensor.shape),tensor.dtype,offsets[name])
            mc_copy_tensor(tensor,out[name])
        torch.cuda.synchronize()
        for name,tensor in weights.items():
            if not torch.equal(tensor.view(torch.uint8),out[name].view(torch.uint8)):
                raise RuntimeError('compressed weight copy changed bytes')
        return out,owner
    except Exception:
        torch.cuda.synchronize();owner.close();raise
'''
    out=HERE/'prepare.py';out.write_text(body)
    report['dropin']=dict(path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(body.encode()).hexdigest(),
        integration='In prepare, pass only immutable tensors; update their state entries with returned views and retain owner. Synchronization counters stay normal. No timed allocation or copy.')
(HERE/'h200_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
