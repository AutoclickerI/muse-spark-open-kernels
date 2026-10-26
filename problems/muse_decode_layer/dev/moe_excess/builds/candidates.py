"""Production S22 counterfactuals for every large-M ladder outcome.

All expert arithmetic and metadata remain timed. No excluded-work oracle is a
candidate. Every arm uses the same original S22 before/after the heavy branch.
"""
import ast
import base64
import difflib
import hashlib
import importlib.util
import json
import lzma
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[4]
P = R / 'problems/muse_decode_layer'
sys.path[:0] = [str(R / 'tools'), str(P / 'dev/s16')]
from pack import pack

BASE_SHA = '4b8c803db080f4e0f2a69e4c666ee91573fc6ffde238d2e63db4223f83f84d27'
BASE_PACKED = 'd1ccdc358ef5fb0f038a1d0e741465f512f498e90d39a16e56f3f0b0e47615f3'
BASE_REVISION = 'daa7a1c3b902e80537e2e50f1c7d6c6ea52fb527'
SPECS = {
    'REG': 'L0 minus STREAM / operand issue: direct packed Q/scale register loads; one warpgroup and activation-only shared ring',
    'DECODE': 'L1 minus L0 / register pressure: single decoded fragment lifetime; synchronous completion at substep reuse',
    'PAD': 'L2 minus L1 / padded MMA: exact 16/32/48/64-row tail widths instead of 32/64',
    'OCC': 'L0 minus STREAM / residency: one warpgroup, 32-row tiles and three stages, 38400B rather than 102400B',
    'BALANCE': 'CTA/SM tail / expert imbalance: flatten expert,row64,channel into uniform grid-stride tasks',
    'EPI': 'L3 minus L2 / epilogue-readiness: hoisted norm/global scales and warp-only PDL finalizer',
}
ROWS = ['v16S', 'v16F', 'v32S', 'v32F']


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def once(s, a, b, count=1):
    assert s.count(a) == count, (a[:90], s.count(a), count)
    return s.replace(a, b)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base():
    # P4's morning selection may overwrite its working S22. Read the exact
    # committed build screened here; no mutable-file fallback or silent rebase.
    packed = subprocess.check_output(['git', 'show', BASE_REVISION +
        ':problems/muse_decode_layer/cand_r3_best7_s22_final.py'], cwd=R)
    assert sha(packed) == BASE_PACKED
    call = next(n for n in ast.walk(ast.parse(packed)) if isinstance(n, ast.Call)
                and getattr(n.func, 'attr', '') == 'b85decode')
    source = lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0]))).decode()
    assert sha(source) == BASE_SHA
    return source


def mma48():
    n = 24
    dst = ','.join('%' + str(i) for i in range(n))
    src = ','.join('%' + str(n + i) for i in range(4))
    constraints = ','.join('"+f"(c[' + str(i) + '])' for i in range(n))
    args = ','.join('"r"(a[' + str(i) + '])' for i in range(4))
    return (' else if constexpr(NT==48){\n asm volatile("{ .reg .pred p;setp.ne.b32 p,1,0;'
            f'wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {{{dst}}},{{{src}}},%28,p,1,1,0;}}"\n'
            f' :{constraints}:{args},"l"(d));\n }}\n')


def variant(original, name):
    a = original.index('namespace chunk {')
    b = original.index('template<int S,bool F>std::vector<int64_t> ch_resource()', a)
    s = original[a:b].replace('namespace chunk {', 'namespace pb_' + name.lower() + ' {', 1)
    if name == 'REG':
        s = once(s, 'SB=512,WG=2;', 'SB=512,WG=1;')
        s = once(s, '__launch_bounds__(256,2)', '__launch_bounds__(128,2)')
        s = once(s, 'S=BR_HOPPER?DEPTH:3,MT=F?64:32;', 'S=BR_HOPPER?DEPTH:3,MT=F?128:64;')
        s = once(s, 'mt*2+(threadIdx.x>>7)', 'mt')
        s = once(s, 'return ((S*WG*(QB+SB)+1023)&~1023)+S*MAX*KT*2;', 'return S*MAX*KT*2;')
        s = once(s, 'BS=(S*WG*(QB+SB)+1023)&~1023;', 'BS=0;')
        start = s.index('  if(t<64){\n', s.index('auto issue='))
        end = s.index('  asm volatile("cp.async.commit_group;', start)
        s = s[:start] + s[end:]
        s = once(s, 'uint4 sv=*(const uint4*)(ss+(st*WG+wg)*SB+(w*8+(l>>2))*16);',
                 'uint4 sv=*(const uint4*)(SC+(((size_t)e*(N/16)+mt*4+(t>>5))*128*32)+((t>>2)&7)*16+k*128);')
        s = once(s, 'uint4 q=*(const uint4*)(qs+(st*WG+wg)*QB+sub*2048+t*16);',
                 'uint4 q=*(const uint4*)(Q+(((size_t)e*(N/16)+mt*4+(t>>5))*128*64)+(t&15)*4+k*256+(sub*2+((t>>4)&1))*64);')
    elif name == 'DECODE':
        # A completed subgroup releases A registers before the next subgroup
        # overwrites them. No async register read is left outstanding.
        s = once(s, 'float acc[NT/2]={};unsigned af[2][16]={};',
                 'float acc[NT/2]={};unsigned af[1][16]={};')
        s = s.replace('af[sub^1]', 'af[0]').replace('af[sub]', 'af[0]')
        s = once(s, 'keep(af[0]);keep(af[1]);', 'keep(af[0]);')
        s = s.replace('wgmma.wait_group.sync.aligned 1;', 'wgmma.wait_group.sync.aligned 0;')
    elif name == 'PAD':
        s = once(s, ' else if constexpr(NT==64){', mma48() + ' else if constexpr(NT==64){')
        s = once(s, 'if(rem<=32){CH_TILE(32);}else{CH_TILE(64);}',
                 'if(rem<=16){CH_TILE(16);}else if(rem<=32){CH_TILE(32);}else if(rem<=48){CH_TILE(48);}else{CH_TILE(64);}')
    elif name == 'OCC':
        s = once(s, 'SB=512,WG=2;', 'SB=512,WG=1;')
        s = once(s, '__launch_bounds__(256,2)', '__launch_bounds__(128,2)')
        s = once(s, 'S=BR_HOPPER?DEPTH:3,MT=F?64:32;', 'S=3,MT=F?128:64;')
        s = once(s, 'for(int row=0;row<n;row+=64){int rem=min(n-row,64);',
                 'for(int row=0;row<n;row+=32){int rem=min(n-row,32);')
        s = once(s, 'mt*2+(threadIdx.x>>7)', 'mt')
        s = once(s, 'if(rem<=32){CH_TILE(32);}else{CH_TILE(64);}', 'CH_TILE(32);')
    elif name == 'BALANCE':
        s = once(s, ' for(int u=blockIdx.x;u<32*MT;u+=gridDim.x){int e=ORDER[u/MT],mt=u%MT,n=C[e];if(n<=16)continue;\n  for(int row=0;row<n;row+=64){int rem=min(n-row,64);',
                 ''' __shared__ int prefix[33];
 if(threadIdx.x==0){int sum=0;prefix[0]=0;for(int i=0;i<32;i++){int n=C[ORDER[i]];sum+=n>16?(n+63)/64:0;prefix[i+1]=sum;}}
 __syncthreads();
 for(int u=blockIdx.x;u<prefix[32]*MT;u+=gridDim.x){int group=u/MT,mt=u%MT,ix=0;
  for(int d=16;d;d>>=1)if(prefix[ix+d]<=group)ix+=d;
  int e=ORDER[ix],n=C[e],row=(group-prefix[ix])*64,rem=min(n-row,64);''')
        s = once(s, '#undef CH_TILE\n  }\n }', '#undef CH_TILE\n }')
    elif name == 'EPI':
        s = once(s, 'int rank=blockIdx.x%CS;',
                 '''int rank=blockIdx.x%CS;
 float early_g=GS[e]*256.f,early_u=F?GS[e+32]*256.f:0.f;
 float early_n0=F?0.f:1.f+__bfloat162float(PNW[mt*64+w*16+(l>>2)]);
 float early_n1=F?0.f:1.f+__bfloat162float(PNW[mt*64+w*16+(l>>2)+8]);''')
        s = once(s, 'float gg=GS[e]*256.f,gu=GS[e+32]*256.f;', 'float gg=early_g,gu=early_u;')
        s = once(s, '}else{float gs=GS[e]*256.f;', '}else{float gs=early_g;')
        s = once(s, '(1.f+__bfloat162float(PNW[cc]))', '(((j>>1)&1)?early_n1:early_n0)')
    return s


HOST = r'''
#define PB_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]
template<int KIND,bool F>std::vector<int64_t> pb_go(std::vector<int64_t> v,int grid,int M,cudaStream_t stream,bool launch){
 int dev,major,resident;cudaFuncAttributes attr{};BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));
 #define PB_GO(K,NS,S,MAX,THREADS) if constexpr(KIND==K){int bytes=NS::dyn<S,MAX>();auto kernel=NS::kernel<S,F>;BR_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));BR_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,THREADS,bytes));if(launch)kernel<<<grid,THREADS,bytes,stream>>>(PB_ARGS(v),M);return {attr.numRegs,(int64_t)attr.localSizeBytes,(int64_t)attr.sharedSizeBytes,bytes,resident};}
 // sm120's arithmetic stand-in uses the original depth3 pipeline.
 if(major==9){PB_NATIVE}else{PB_LOCAL}
 #undef PB_GO
 return {};
}
std::vector<int64_t> night_run(std::vector<int64_t> v,int64_t first,int64_t kind,int64_t grid,int64_t M,int64_t stream,bool launch){
 #define PB_CASE(K) case K:return first?pb_go<K,true>(v,grid,M,(cudaStream_t)stream,launch):pb_go<K,false>(v,grid,M,(cudaStream_t)stream,launch);
 switch(kind){PB_CASE(0) PB_CASE(1) PB_CASE(2) PB_CASE(3) PB_CASE(4) PB_CASE(5)}
 #undef PB_CASE
 throw std::runtime_error("night MoE kind");
}
'''


TAIL = '''
P3B_NIGHT_MODULE=[]
def p3b_night_module():
 if not P3B_NIGHT_MODULE:
  target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
  mod=load_inline(name='p3b_night_'+hashlib.sha256(P3B_NIGHT_CUDA.encode()).hexdigest()[:12],cpp_sources=P3B_NIGHT_CPP,cuda_sources=P3B_NIGHT_CUDA,functions=['night_run'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target])
  P3B_NIGHT_MODULE.append(mod)
 return P3B_NIGHT_MODULE[0]
_p3b_night_original_plan=p3b_plan
'''


def main():
    (H / 'out').mkdir(exist_ok=True)
    source = base()
    tree = ast.parse(source)
    original = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], 'id', '') == 'P3B_CUDA')
    additions = '\n'.join(variant(original, name) for name in SPECS)
    native = []
    local = []
    for k, name in enumerate(SPECS):
        ns = 'pb_' + name.lower()
        nt, threads = (32, 128) if name == 'OCC' else (64, 128 if name == 'REG' else 256)
        for cases, depth in ((native, 3 if name == 'OCC' else 4), (local, 3)):
            cases.append(f'PB_GO({k},{ns},{depth},{nt},{threads})')
    cuda = original + '\n' + additions + HOST.replace('PB_NATIVE', ' '.join(native)).replace('PB_LOCAL', ' '.join(local))
    cpp = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nstd::vector<int64_t> night_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,bool);'
    (H / 'out/night.cu').write_text(cuda)
    common = '\nP3B_NIGHT_CUDA=' + repr(cuda) + '\nP3B_NIGHT_CPP=' + repr(cpp) + '\n' + TAIL
    plan = next(ast.get_source_segment(source, n) for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == 'p3b_plan')
    plan = once(plan, 'light_grid=4;tail=mt_module();pm=mt_pmz_module()',
                'light_grid=4;tail=mt_module();pm=mt_pmz_module();night=p3b_night_module()')
    for vector, first in (('p13', 1), ('p2', 0)):
        plan = once(plan, f'mod.ch_run({vector},{first},4,sms*4,m,side.cuda_stream)',
                    f'night.night_run({vector},{first},P3B_NIGHT_KIND,sms*4,m,side.cuda_stream,True)')
    specs = {}
    base_lines = source.splitlines(True)
    for k, (name, mechanism) in enumerate(SPECS.items()):
        tail = plan
        if name == 'EPI':
            tail = once(tail, 'tail.mt_finish(fp,m,start,10,', 'tail.mt_finish(fp,m,start,14,')
        text = source + common + '\nP3B_NIGHT_KIND=' + str(k) + '\n' + tail + '\n'
        compile(text, '<p3b-' + name + '>', 'exec')
        (H / 'out' / (name + '.py')).write_text(text)
        shipped = '\n'.join(line.rstrip() for line in pack(text).splitlines()) + '\n'
        (H / (name + '.py')).write_text(shipped)
        edits = [(a, b, ''.join(text.splitlines(True)[c:d])) for op, a, b, c, d in
                 difflib.SequenceMatcher(None, base_lines, text.splitlines(True), autojunk=False).get_opcodes()
                 if op != 'equal']
        specs[name] = dict(sha256=sha(text), shipped_sha256=sha(shipped), edits=edits,
                           rows=ROWS, mechanism=mechanism, kind=k)
    (H / 'out/base.py').write_text(source)
    (H / 'out/arms.json').write_text(json.dumps(specs))
    (H / 'candidates.json').write_text(json.dumps(dict(base_sha256=sha(source),
        base_file='problems/muse_decode_layer/cand_r3_best7_s22_final.py',
        native_cuda_sha256=sha(cuda), arms={k: {a: b for a, b in v.items() if a != 'edits'}
                                           for k, v in specs.items()}), indent=2) + '\n')
    print('built', list(specs), 'on S22', sha(source)[:12], 'CUDA', sha(cuda)[:12])


if __name__ == '__main__':
    main()
