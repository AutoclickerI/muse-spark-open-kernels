"""Freeze NT1 controls, phase instrumentation and exact decode/prefetch experiments."""
import ast
import hashlib
from pathlib import Path
import re

from build_moe_real_routing import namespace
from build_moe_small_debug import select
from build_moe_v9_reuse import reference_generator

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / 'problems/muse_decode_layer/dev/nt1_limit'
HERE = Path(__file__).resolve().parent


def literal(source, name):
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)


def thin(source, trace=False):
    source = source.replace('#include <torch/extension.h>', '#include <cuda_runtime.h>\n#include <stdexcept>\n#include <vector>')
    source = source.replace('#include <ATen/cuda/CUDAContext.h>', '')
    pre = '''
#define C10_CUDA_CHECK(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
'''
    source = pre + source[:source.index('\nvoid run(')]
    extra = 'v[11]?(float*)v[11]:nullptr,v[12]?(int64_t*)v[12]:nullptr,' if trace else ''
    ci, oi = (13, 14) if trace else (11, 12)
    source += '''
void run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream){
 moe_fused<1><<<grid,NTHR,smem_bytes(1,stages,M),(cudaStream_t)stream>>>(
 (const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],
 (const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],
 (__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],''' + extra + f'''(int*)v[{ci}],(__nv_bfloat16*)v[{oi}],M,start,stages);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
'''
    assert 'torch::' not in source and 'at::' not in source
    return source


def scale16(source):
    source = source.replace('TILE_BYTES = 36864', 'TILE_BYTES = 40960')
    old = '''const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);
        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};'''
    new = '''const uint4 scv0 = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + (warp * 32 + lane)*2);
        const uint4 scv1 = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + (warp * 32 + lane)*2+1);
        const uint32_t scw[8] = {scv0.x,scv0.y,scv0.z,scv0.w,scv1.x,scv1.y,scv1.z,scv1.w};'''
    assert old in source; source = source.replace(old, new)
    source = source.replace('const uint32_t sw = scw[g >> 1];', 'const uint32_t sw = scw[g];')
    source = source.replace('__byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u',
                            '__byte_perm(sw, 0u, 0x1010u)')
    source = source.replace('__byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u',
                            '__byte_perm(sw, 0u, 0x3232u)')
    return source


def early(source):
    source = source.replace('uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);',
        '''__nv_bfloat16* bcache = reinterpret_cast<__nv_bfloat16*>(red + NCW*32*RED);
  uint64_t* fullb = reinterpret_cast<uint64_t*>(bcache + NCW*4*8*64);''')
    source = source.replace('2 * NCW * 32 * nt * 4 * sizeof(float)', '(NCW*32*nt*4*sizeof(float)+32768)')
    source = source.replace('uint32_t Bf[NT][8][8];', 'uint32_t Bf[NT][4][8];')
    old = '''Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;
            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;'''
    new = '''if(g<4){
              uint4* bp=(uint4*)(bcache+(warp*4+g)*8*64+(lane>>2)*64+q*16);
              bp[0]=v0;bp[1]=v1;
            }else{
              Bf[j][g-4][0]=v0.x;Bf[j][g-4][1]=v0.y;Bf[j][g-4][2]=v0.z;Bf[j][g-4][3]=v0.w;
              Bf[j][g-4][4]=v1.x;Bf[j][g-4][5]=v1.y;Bf[j][g-4][6]=v1.z;Bf[j][g-4][7]=v1.w;
            }'''
    assert old in source;source=source.replace(old,new)
    load = '''__device__ __forceinline__ uint4 load4_shared(const void* p){
 uint4 v;asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(smem_u32(p)):"memory");return v;
}
'''
    source=source.replace('struct PrefetchState {',load+'\nstruct PrefetchState {')
    old='''const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};'''
    new=old+'''
        uint4 prefetched[8];
        #pragma unroll
        for(int g=0;g<8;++g)prefetched[g]=load4_shared(wq+g*32);
        __syncwarp();if(lane==0)mbar_arrive(&emptyb[st]);bar_consumers();
        if(tid==0)prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
'''
    assert old in source;source=source.replace(old,new)
    source=source.replace('const uint4 wv = wq[g * 32];','''const uint4 wv = prefetched[g];
          uint32_t bc[8];
          if(g<4){
            const uint4* bp=(const uint4*)(bcache+(warp*4+g)*8*64+(lane>>2)*64+q*16);
            uint4 b0=load4_shared(bp),b1=load4_shared(bp+1);
            bc[0]=b0.x;bc[1]=b0.y;bc[2]=b0.z;bc[3]=b0.w;
            bc[4]=b1.x;bc[5]=b1.y;bc[6]=b1.z;bc[7]=b1.w;
          }else{
            #pragma unroll
            for(int k=0;k<8;++k)bc[k]=Bf[0][g-4][k];
          }''')
    source=source.replace('Bf[j][g][2 * s], Bf[j][g][2 * s + 1]', 'bc[2*s],bc[2*s+1]')
    source=source.replace('if (lane == 0) mbar_arrive(&emptyb[st]);','')
    source=source.replace('if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);','')
    source=source.replace('buf ^= 1;', 'bar_consumers(); // protect the single reduction buffer')
    return source


def diagnostic(source, mode):
    if mode=='no_decode':
        a=source.index('          const uint4 wv = wq[g * 32];')
        b=source.index('#pragma unroll\n          for (int s = 0;',a)
        source=source[:a]+'''          const uint32_t Sr=0x3f803f80u,Sr8=Sr;uint32_t o[16];
          #pragma unroll
          for(int k=0;k<16;++k)o[k]=0x3f803f80u;
'''+source[b:]
    elif mode=='no_mma':
        source=re.sub(r'for \(int j = 0; j < NT; \+\+j\) mma16816\([^;]+;',
            'for(int j=0;j<NT;++j){asm volatile(""::"r"(a0),"r"(a1),"r"(a2),"r"(a3):"memory");}',source)
    else:raise ValueError(mode)
    return source


def lop3(source):
    old='const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);'
    new='uint32_t W; asm("lop3.b32 %0,%1,%2,%3,0xe4;":"=r"(W):"r"(A),"r"(B),"n"(LA_MASK));'
    assert old in source
    return source.replace(old,new)


def main():
    DEV.mkdir(exist_ok=True,parents=True)
    orig=(ROOT/'problems/muse_moe_w4a16/cand_v9.py').read_text()
    raw=(ROOT/'problems/muse_decode_layer/cand_r3_best6_small_nt2.py').read_text()
    phase=(ROOT/'problems/muse_moe_w4a16/cand_v9_phase.py').read_text()
    native=literal(orig,'CUDA_SRC')
    sources={'recode':thin(native),'raw':thin(literal(raw,'V9_CUDA_SRC')),
             'trace':thin(literal(phase,'CUDA_SRC'),True),'scale16':thin(scale16(native)),
             'early':thin(early(native)),'lop3':thin(lop3(native)),
             'no_decode':thin(diagnostic(native,'no_decode')),'no_mma':thin(diagnostic(native,'no_mma'))}
    for name,source in sources.items():(DEV/(name+'.cu')).write_text(source)
    names={'H','NLOC','TOPK','TILE_BYTES','NT1','NCNT','_DQ_SLOTS','_LA_MASK','_bit_table','_TABLE','_encode_scales','_repack'}
    helpers=select(orig,names)
    raw_repack=select(raw,{'_repack'}).replace('def _repack(', 'def _raw_repack(')
    fixtures=(ROOT/'problems/muse_decode_layer/moe_tune.py').read_text();fixtures=fixtures[:fixtures.index('\ndef _graph_time')]
    old=(HERE/'moe_real_routing_driver.py').read_text()
    graph=select(old,{'LabGraph','lab_pair'})
    prefix='import torch\nfrom torch.utils.cpp_extension import load_inline\nimport time as lab_time\nimport statistics as lab_stats\nlab_torch=torch\n'
    body=prefix+'\nNT1_SOURCES='+repr(sources)+'\n'+helpers+raw_repack+'\n'+namespace(fixtures,'fx_')+reference_generator()+graph
    body+=(DEV/'driver.py.inc').read_text()
    out=ROOT/'prof/lab/moe_nt1_limit.py';out.write_text(body)
    driver=(DEV/'driver.py.inc').read_text()
    drop=select(driver,{'NT1_CPP','NT1_MODULES','NT1_HOPPER','nt1_module','nt1_call'})
    prepare='''
def prepare_kind(weights,kind='lop3'):
    if kind not in ('recode','raw','lop3','early'):raise ValueError(kind)
    nt1_module(kind)
    device=weights['w13'].device
    r=torch.arange(512,device=device)[:,None]*8+torch.arange(8,device=device)
    rows={'w13':torch.cat((r,r+4096),1),'w2':torch.arange(4096,device=device).view(256,16)}
    bank={}
    for name,row in rows.items():
        pieces=[(_raw_repack if kind=='raw' else _repack)(weights[name][e],weights[name+'_sf'][e],row) for e in range(32)]
        value=torch.stack(pieces)
        bank[name+'p']=value
    return dict(banks={kind:bank},counters={kind:torch.zeros(64,device=device,dtype=torch.int32)},
      gs13=weights['w13_gs'].float().contiguous(),gs2=weights['w2_gs'].float().contiguous(),
      pnw=weights.get('post_norm_w',weights.get('post_expert_norm')).contiguous(),
      sms=torch.cuda.get_device_properties(device).multi_processor_count)
'''
    (DEV/'dropin.py').write_text(prefix+'\nimport hashlib\nNT1_SOURCES='+repr({k:v for k,v in sources.items() if k in ('recode','raw','early','lop3')})+'\n'+helpers+raw_repack+drop+prepare)
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest())


if __name__=='__main__':main()
