"""Native int4 and cp.async reconstruction, bit for bit in all three layouts."""
import importlib.util,hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('dc_readers',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
src=(H/'native.cuh').read_text()+r'''
__global__ void check(const unsigned char* w,int4*out,int staged){
 const int lane=threadIdx.x,unit=blockIdx.x;int4 words[3];
 __shared__ int4 sm[96];
 if(staged){dc_stage(w,unit,lane,sm+lane);asm volatile("cp.async.commit_group;":::"memory");asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();words[0]=sm[lane];words[1]=sm[lane+32];int4 z=sm[lane+64];dc_expand(z.x,z.y,z.z,z.w,((const unsigned*)w)[4],((const unsigned*)w)[12],words+2);}
 else dc_read(w,unit,lane,words);
 for(int j=0;j<3;++j)out[unit*96+j*32+lane]=words[j];
}
void launch(uint64_t w,uint64_t out,int u,int staged,uint64_t stream){check<<<u,32,0,(cudaStream_t)stream>>>((const unsigned char*)w,(int4*)out,staged);}
__global__ void check_point(const void*w,__nv_bfloat16*out,int n,int k){int i=blockIdx.x*256+threadIdx.x;if(i<n*k)out[i]=dc_point(w,i/k,i%k);}
void launch_point(uint64_t w,uint64_t out,int n,int k,uint64_t stream){check_point<<<(n*k+255)/256,256,0,(cudaStream_t)stream>>>((const void*)w,(__nv_bfloat16*)out,n,k);}
'''
decl='void launch(uint64_t,uint64_t,int,int,uint64_t);void launch_point(uint64_t,uint64_t,int,int,uint64_t);'
arch=''.join(map(str,torch.cuda.get_device_capability()));tag=hashlib.sha256(src.encode()).hexdigest()[:12]
mod=load_inline(name='dc_reader_proof_'+tag+'_'+arch,cpp_sources=decl,cuda_sources=src,functions=['launch','launch_point'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
g=torch.Generator(device='cuda').manual_seed(61791);rows=[]
for fmt in range(3):
    for mode in ('uniform','sparse'):
        raw=torch.randint(0,256,(1024*1536,),device='cuda',generator=g,dtype=torch.uint8)
        if mode=='sparse':
            # This forces many absent extension lanes and occasional present
            # ones, independently of a particular BF16 distribution.
            raw.view(-1,3,32,16)[:,2]&=17
            raw.view(-1,3,32,16)[::97,2,::7]=255
        bank,_=m.dc_pack(raw,fmt,n=16384,k=64);out=torch.empty_like(raw)
        for staged in (0,1):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):mod.launch(bank.data_ptr(),out.data_ptr(),1024,staged,torch.cuda.current_stream().cuda_stream)
            graph.replay();torch.cuda.synchronize();assert torch.equal(raw,out),(fmt,mode,staged)
            rows.append(dict(fmt=fmt,distribution=mode,cp_async=bool(staged),exact=True));print('PASS',rows[-1],flush=True)
for n,k in ((8192,1024),(256,8192)):
    raw=(torch.randn((n,k),device='cuda',generator=g)/(k**.5)).bfloat16()
    raw[3,139]=0.;raw[11,211]=-0.;raw[19,27]=3.;raw[23,39]=torch.finfo(torch.bfloat16).tiny
    c,_=m.dc_encode_cd(m.cd_encode(raw));out=torch.empty_like(raw);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):mod.launch_point(c['coded'].data_ptr(),out.data_ptr(),n,k,torch.cuda.current_stream().cuda_stream)
    graph.replay();torch.cuda.synchronize();assert torch.equal(raw.view(torch.int16),out.view(torch.int16)),(n,k)
    rows.append(dict(reader='CUDA point (O32 shared producer)',n=n,k=k,exact=True));print('PASS',rows[-1],flush=True)
(H/'reader_validation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),passed=True,rows=rows),indent=2)+'\n')
