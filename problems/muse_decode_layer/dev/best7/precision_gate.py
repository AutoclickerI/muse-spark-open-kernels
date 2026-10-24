"""Check stored-scale banks and timed expansion over every finite E4M3 code."""
from pathlib import Path
import argparse,hashlib, importlib.util, json
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,default=ROOT/'problems/muse_decode_layer/cand_r3_best7.py')
parser.add_argument('--out',default='precision_gate_local.json');args=parser.parse_args()
source = args.candidate
spec = importlib.util.spec_from_file_location('best7_precision', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
device = 'cuda'
codes = torch.arange(256, device=device, dtype=torch.uint8)
codes = codes[(codes & 127) != 127]
sf = codes.repeat((16*256+codes.numel()-1)//codes.numel())[:16*256].reshape(16,256).contiguous().view(torch.float8_e4m3fn)
wq = (torch.arange(16*2048, device=device).reshape(16,2048)%256).to(torch.uint8)
rows = torch.arange(16, device=device).reshape(1,16)
bank = m._pf_repack(wq,sf,rows)
assert torch.equal(bank,m._repack(wq,sf,rows)), 'optional canonical bank differs from byte-preserving V9'
out = torch.empty((1,16,4096),device=device,dtype=torch.bfloat16)
counts = torch.ones(1,device=device,dtype=torch.int32)
m._pf_canonical_expand_coalesced[(1,1,4)](bank,bank.view(torch.int32),out.view(torch.int32),counts,
    ROWS=16,GATED=False,KB=1024,num_warps=8)
lut = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device=device)
weight = lut[torch.stack((wq&15,wq>>4),-1).long()].reshape(16,4096)
expected = (weight*sf.float().repeat_interleave(16,-1)).bfloat16()
assert torch.equal(out[0],expected), 'timed DG/Humming-stand-in expansion changed raw scale values'

wq = wq.repeat(16,1).unsqueeze(0)
sf = sf.repeat(16,1).unsqueeze(0)
blob,raw_sf = m._pf_raw_repack(wq,sf,False)
assert torch.equal(raw_sf.flatten().sort().values,sf.view(torch.uint8).flatten().sort().values)
encoded = torch.empty_like(blob)
m._pf_encode_blob_fast[(blob.numel()//2048,)](blob.view(torch.int32),raw_sf,encoded.view(torch.int32),num_warps=4)
words = encoded.view(torch.int32).reshape(-1,2).long() & 0xffffffff
recovered = torch.zeros((words.shape[0],16),device=device,dtype=torch.int64)
for k,h,b,word,p in m._pf_TABLE:
    recovered[:,k*2+h] |= ((words[:,0 if word=='A' else 1]>>p)&1)<<b
original = blob.view(torch.int32).reshape(-1,2).long() & 0xffffffff
want = ((original[:,:,None]>>torch.arange(0,32,4,device=device))&15).reshape(-1,16)
assert torch.equal(recovered,want), 'raw17 bit permutation folded scale sign/zero into NVFP4 codes'
result=dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),finite_scale_codes=codes.numel(),
    canonical_raw_bank_exact=True,timed_expansion_exact=True,raw17_scale_bytes_preserved=True,
    raw17_nvfp4_codes_recovered_exactly=True)
(HERE/args.out).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
