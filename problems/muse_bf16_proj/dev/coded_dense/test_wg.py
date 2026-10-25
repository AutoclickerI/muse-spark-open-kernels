"""Exercise the actual TMA activation layout and lossless WGMMA fragment ownership on sm_120."""
import hashlib
import json
from pathlib import Path
import torch
from coded import cd_module
from wg_frozen import wg_encode,wg_decode,wg_gemv,wg_module

H=Path(__file__).resolve().parent

def main():
    torch.manual_seed(61791)
    cd_module(((1,1,2,3),))
    wg_module()
    exact=[];gemvs=[]
    g=torch.Generator(device='cuda').manual_seed(61791)
    pats=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256,256)
    for lay in (False,True):
        c=wg_encode(pats,kmajor=lay)
        assert torch.equal(wg_decode(c).view(torch.int16),pats.view(torch.int16))
        exact.append(dict(case='all bf16 bit patterns',kmajor=lay))
    for N,K in ((2304,8192),(4608,8192),(8192,1024),(1024,4096),(8192,4096)):
        w=(torch.randn((N,K),device='cuda',generator=g)/K**.5).to(torch.bfloat16)
        # Exact finite escapes, zeros and signed zero exercise address inversion too.
        w[3,7]=0.;w[11,139]=-0.;w[51,27]=3.
        for lay in (False,True):
            c=wg_encode(w,kmajor=lay)
            assert torch.equal(wg_decode(c).view(torch.int16),w.view(torch.int16))
            exact.append(dict(case=f'{N}x{K}',kmajor=lay))
            for M in (1,5,8,12,16,20,32):
                x=torch.randn((M,K),device='cuda',generator=g).to(torch.bfloat16)
                ref=x.double()@w.double().T
                for S in (2,4,8,16):
                    if K%(128*S):continue
                    MT=max(8,1<<(M-1).bit_length());smem=K//64//S*MT*128+K//64//S*2
                    if smem>torch.cuda.get_device_properties(0).shared_memory_per_block_optin:continue
                    for NW in (4,8):
                        for WB in (1,2):
                            p,_=wg_gemv(x,c,S,NW=NW,WB=WB,pdl=False)
                            err=float((p.double().sum(0)-ref).norm()/ref.norm())
                            assert err<3e-5,(N,K,M,S,NW,WB,lay,err)
                            gemvs.append(dict(N=N,K=K,M=M,S=S,NW=NW,WB=WB,kmajor=lay,error=err))
                # Capture the host tensor-map ABI, real bulk copies and all epilogue logic.
                for WB in (1,2):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):p,_=wg_gemv(x,c,8,NW=4,WB=WB,pdl=True)
                    graph.replay();torch.cuda.synchronize()
                    assert float((p.double().sum(0)-ref).norm()/ref.norm())<3e-5
        print(f'PASS {N}x{K}',flush=True)
    out=dict(passed=True,wg_kernel_sha256=hashlib.sha256((H/'wg_kernel.cu').read_bytes()).hexdigest(),
             exact_cases=len(exact),gemv_cases=len(gemvs),max_error=max(r['error'] for r in gemvs),
             actual_tma_layout=True,arithmetic='mma.sync stand-in for sm90 WGMMA',exact=exact)
    (H/'wg_local.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)

if __name__=='__main__':main()
