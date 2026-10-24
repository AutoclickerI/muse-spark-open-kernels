"""Build W13 timing-only ablations of the exact emitted TileLang kernel; never a candidate."""
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from engine import mw_compile


def main():
    includes=set();parts=[];cases=[];launch=[]
    for m in (128,256):
        for kind,grid in (('base',132),('base',264),('base',528),('reinterpret',264),('constant',264),('no_mma',264)):
            code=mw_compile(m,8192,BN=32,GRID=grid,target='sm_90a').get_kernel_source()
            includes.update(re.findall(r'^#include .*$',code,re.M));code=re.sub(r'^#include .*$', '',code,flags=re.M)
            index=len(cases);ns=f'ab{index}';kernel=f'ab_kernel_{index}'
            code=code.replace('main_kernel',kernel)
            if kind in ('reinterpret','constant'):
                name='mw_reinterpret' if kind=='reinterpret' else 'mw_constant'
                code=code.replace('= mw_decode(',f'= {name}(')
                helper=('return cutlass::bfloat16_t::bitcast((unsigned short)(q>>(4*odd)));'
                        if kind=='reinterpret' else 'return cutlass::bfloat16_t(1.f);')
                code='__device__ __forceinline__ cutlass::bfloat16_t '+name+'(unsigned q,unsigned s,int odd){'+helper+'}\n'+code
            if kind=='constant':
                code=re.sub(r'^.*tl::cp_async_gs_conditional<\d+>.*\b(?:qs|ss)\b.*$', '',code,flags=re.M)
            if kind=='no_mma':
                code=re.sub(r'^.*tl::wgmma_rs<.*$', '',code,flags=re.M)
                body=code.index('{',code.index('__launch_bounds__'))
                code=code[:body+1]+'\n unsigned ab_sink=0;\n'+code[body+1:]
                code=code.replace('tl::warpgroup_arrive();',
                    '#pragma unroll\n for(int ab_i=0;ab_i<32;++ab_i)ab_sink^=reinterpret_cast<const uint32_t*>(af)[ab_i];\n'
                    'tl::warpgroup_arrive();')
                end=code.rfind('}')
                code=code[:end]+f'\n reinterpret_cast<unsigned*>(Y)[{(m*8+1024-16)*8192}+blockIdx.x*128+threadIdx.x]=ab_sink;\n'+code[end:]
            parts.append(f'namespace {ns} {{\n'+code+'\n}\n')
            launch.append(f'case {index}: {ns}::{kernel}<<<{grid},128,17408,stream>>>(c,g,o,r,q,s,l,t,x,y);break;')
            cases.append((m,kind,grid))
    wrapper=r'''
int64_t ab_hostcheck(std::vector<torch::Tensor> v,int64_t variant){
 TORCH_CHECK(v.size()==10&&variant>=0&&variant<12);
 TORCH_CHECK(v[0].scalar_type()==torch::kUInt8&&v[1].scalar_type()==torch::kUInt8);
 TORCH_CHECK(v[2].scalar_type()==torch::kFloat32&&v[3].scalar_type()==torch::kBFloat16);
 for(int i=4;i<9;++i)TORCH_CHECK(v[i].scalar_type()==torch::kInt32);
 TORCH_CHECK(v[9].scalar_type()==torch::kFloat32&&v[0].size(1)==8192&&v[3].size(1)==4096);
 return variant;
}
void ab_run(std::vector<torch::Tensor> v,int64_t variant){
 TORCH_CHECK(v.size()==10);auto stream=at::cuda::getCurrentCUDAStream();
 auto*q=(const unsigned char*)v[0].data_ptr();auto*s=(const unsigned char*)v[1].data_ptr();
 auto*g=(const float*)v[2].data_ptr();auto*x=(const cutlass::bfloat16_t*)v[3].data_ptr();
 auto*c=(const int*)v[4].data_ptr();auto*o=(const int*)v[5].data_ptr();auto*r=(const int*)v[6].data_ptr();
 auto*t=(const int*)v[7].data_ptr();auto*l=(const int*)v[8].data_ptr();auto*y=(float*)v[9].data_ptr();
 switch(variant){ CASES default:TORCH_CHECK(false,"bad ablation index");}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''.replace('CASES','\n'.join(launch))
    cuda='#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n'+'\n'.join(sorted(includes))+'\n'+''.join(parts)+wrapper
    output=ROOT/'prof/lab/moe_w13_ablation.py'
    old=(ROOT/'prof/lab/moe_mixed_wgmma_planes.py').read_text()
    old=old[:old.index('\ndef lab(deadline,weights,state=None):')]
    # The Python engine and fixture/reference helpers are inlined, but none of
    # the old full-MoE lab is invoked. Wrong-value variants stay prepare-only.
    output.write_text(old+'\nAB_CUDA='+repr(cuda)+'\nAB_CASES='+repr(cases)+'\n'+(HERE/'ablation_driver.py.inc').read_text())
    (HERE/'diagnostics/ablation.cu').write_text(cuda)
    (HERE/'diagnostics/ablation_cases.json').write_text(json.dumps(cases)+'\n')
    print(output,hashlib.sha256(output.read_bytes()).hexdigest(),len(cuda))


if __name__=='__main__':main()
