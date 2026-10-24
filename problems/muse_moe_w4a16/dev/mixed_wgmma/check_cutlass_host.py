"""Reproduce the original host TMA abort and check the repaired geometry on SM120.

No mixed GEMM is launched. The fixed collective remains quarantined pending a
numeric H200 lab; this check tests real CUDA TMA descriptors on the local GPU.
"""
import argparse
import ast
import base64
import hashlib
import resource
import subprocess
import sys
import zlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
ORIGINAL_REV='313a282b8d1393bfdad928716becb53986f4dd1a'
HOST=r'''
int64_t mixed_hostcheck(torch::Tensor a,torch::Tensor s,torch::Tensor shapes,bool host){
 using namespace omni_mixed;int g=a.size(0),m=a.size(1),k=a.size(2),n=s.size(2);
 TORCH_CHECK(s.dim()==3&&s.size(0)==g&&s.size(1)==k/16&&s.is_contiguous()&&n%16==0,"scale shape/layout");
 std::vector<Shape<int,int,int>> hs(g,Shape<int,int,int>{m,n,k});
 ShapeG ps;ps.num_groups=g;ps.problem_shapes=(Shape<int,int,int>*)shapes.data_ptr();ps.host_problem_shapes=host?hs.data():nullptr;
 Main::Arguments ma{};ma.ptr_A=(const EA**)a.data_ptr();ma.ptr_B=(const cutlass::float_e2m1_t**)a.data_ptr();
 ma.ptr_S=(const cutlass::float_e4m3_t**)s.data_ptr();ma.chunk_size=16;
 TORCH_CHECK(Main::can_implement(ps,ma),"can_implement rejected");
 Gemm::Arguments ga{};ga.mode=cutlass::gemm::GemmUniversalMode::kGrouped;
 ga.problem_shape=ps;ga.mainloop=ma;
 ga.epilogue.dC={int64_t(n),Int<1>{},int64_t(0)};ga.epilogue.dD=ga.epilogue.dC;
 TORCH_CHECK(Gemm::can_implement(ga)==cutlass::Status::kSuccess,"grouped Gemm::can_implement rejected");
 auto params=Main::to_underlying_arguments(ps,ma,nullptr);
 return params.scale_k;
}
'''


def child(kind):
    path=ROOT/'problems/muse_decode_layer/dev/omni/mixed.cu'
    source=(subprocess.run(['git','show',ORIGINAL_REV+':problems/muse_decode_layer/dev/omni/mixed.cu'],
                           cwd=ROOT,capture_output=True,text=True,check=True).stdout
            if kind=='original' else path.read_text())+HOST
    mod=load_inline(name='mw_tma_'+kind+'_'+hashlib.sha256(source.encode()).hexdigest()[:10],
        cpp_sources='int64_t mixed_hostcheck(torch::Tensor,torch::Tensor,torch::Tensor,bool);',cuda_sources=source,
        functions=['mixed_hostcheck'],extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/deep_gemm/include'],
        extra_cflags=['-O1','-std=c++20'],extra_cuda_cflags=['-O1','-std=c++20','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
    a=torch.empty((32,128,4096),device='cuda',dtype=torch.bfloat16)
    s=torch.empty((32,256,8192),device='cuda',dtype=torch.float8_e4m3fn)
    cases=[('all_nonempty',torch.full((32,),8)),('mixed_empty',torch.arange(32)%5),('all_empty',torch.zeros(32))]
    tree=ast.parse((ROOT/'prof/lab/moe_vector_wgmma_seed.py').read_text())
    raw=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='CAPTURED_R0' for t in n.targets))
    raw=zlib.decompress(base64.b64decode(raw))
    for m,off in ((128,8*(8+16+32)),(256,8*(8+16+32+128))):
        ids=torch.tensor(list(raw[off:off+8*m]));counts=torch.bincount(ids[ids<32],minlength=32)
        assert (int(counts.sum()),int((counts>0).sum()),int(counts.max()))==({128:(209,18,55),256:(414,21,119)}[m])
        cases.append((f'R0_M{m}_P{int(counts.sum())}_E{int((counts>0).sum())}',counts))
    for label,counts in cases:
        shapes=torch.stack((counts.clamp_min(1),torch.full((32,),8192),torch.full((32,),4096)),1).int().cuda()
        print('HOST_CASE',kind,label,flush=True)
        k=mod.mixed_hostcheck(a,s,shapes,kind=='fixed')
        assert k==256,k
        print('HOST_PASS',label,k,flush=True)
    try:mod.mixed_hostcheck(a,s[:,:,:1].contiguous(),shapes,True)
    except RuntimeError:print('MALFORMED_SCALE_REJECTED',flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    ap=argparse.ArgumentParser();ap.add_argument('--child',choices=['original','fixed']);args=ap.parse_args()
    if args.child:return child(args.child)
    for kind in ('original','fixed'):
        p=subprocess.run([sys.executable,__file__,'--child',kind],capture_output=True,text=True)
        (HERE/'diagnostics'/f'cutlass_{kind}_host.log').write_text(p.stdout+p.stderr)
        print(kind,'returncode',p.returncode,'tail', (p.stdout+p.stderr)[-1400:],flush=True)
        if kind=='original':assert p.returncode!=0 and 'make_tma_copy_desc' in p.stderr
        else:assert p.returncode==0


if __name__=='__main__':main()
