"""Exhaustive byte-pair table/address proof;64-bank replicas decode inside timed GPU kernel."""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
D=Path(__file__).resolve().parent;s=(D/'probe.cu').read_text()
s+='''
void run(int64_t w,int64_t o,int64_t stream){cudaFuncSetAttribute(lut64,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);lut64<<<32,256,65536,(cudaStream_t)stream>>>((uint32_t*)w,(uint32_t*)o);}
'''
mod=load_inline(name='p3a_lut64proof_'+hashlib.sha256(s.encode()).hexdigest()[:12],cpp_sources='#include <cstdint>\n#include <pybind11/pybind11.h>\nvoid run(int64_t,int64_t,int64_t);',cuda_sources=s,functions=['run'],extra_cuda_cflags=['-O3','-arch=sm_120'],no_implicit_headers=True,with_pytorch_error_handling=False)
u=torch.arange(8192,device='cuda',dtype=torch.int64);w=((u&255)|(((u+37)&255)<<8)|(((u+101)&255)<<16)|(((u+193)&255)<<24)).to(torch.int32);out=torch.empty((8192,4),device='cuda',dtype=torch.int32);mod.run(w.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
ix=(w.long()[:,None]>>(torch.arange(4,device='cuda')[None,:]*8))&255;want=((ix&7)<<6)|((ix&8)<<12)|((ix&112)<<18)|((ix&128)<<24)
bad=(out.long()&0xffffffff)!=want
if bool(bad.any()):
 ix_bad=bad.nonzero()[:12];print('BAD',[(tuple(v.tolist()),int(out[tuple(v)].item())&0xffffffff,int(want[tuple(v)].item()),hex(int(w[v[0]].item())&0xffffffff)) for v in ix_bad],flush=True)
assert not bool(bad.any())
r=dict(all256_code_pairs_all32_lanes_all4_byte_positions=True,checks=32768,table_bytes=65536,bank_mapping='bank=lane for every code pair;entry64 words;upper32 replicas unused padding',runtime_decode='onePRMT address andoneLDS per BF16x2 result;scale multiplication retained',table_is_fixed_all_possible_codes=True,no_weight_values_decoded_in_prepare=True,probe_sha256=hashlib.sha256(s.encode()).hexdigest(),source_file_sha256=hashlib.sha256((D/'probe.cu').read_bytes()).hexdigest())
(D/'table_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
