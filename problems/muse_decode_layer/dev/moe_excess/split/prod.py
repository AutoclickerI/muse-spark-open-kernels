"""Production S22 candidates for the separated large-M load parts (whole-load judged by prod_lab.py).

Each arm patches the literal P3B_CUDA / MT_PMZ_CUDA sources of the committed S22 build in place, so
the exact production kernels, grids, streams and shapes are kept; only M128/M256 (v16, v32) use them.
"""
import ast, difflib, importlib.util, json, sys
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[4]
sys.path[:0] = [str(R / 'tools'), str(R / 'problems/muse_decode_layer/dev/s16')]
from pack import pack
# Loaded under a unique name: S22's local stand-ins import unrelated modules called `build`.
_spec = importlib.util.spec_from_file_location('p3b_split_build', H / 'build.py')
_split = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_split)
s22_source, sha, once, balanced = _split.s22_source, _split.sha, _split.once, _split.balanced

ROWS = ['v16S', 'v16F', 'v32S', 'v32F']
PF_ASM = 'asm volatile("cp.async.bulk.prefetch.L2.global [%0],%1;"::"l"({p}),"r"({n}):"memory");'


def pf_heavy(cu):
    # One thread per 16-row group prefetches its contiguous Q (1 KiB/stage) and scale (128 B/stage)
    # rows into L2, stages [0,12) at a tile's first issue, then 8-12 stages ahead of the cp.async ring.
    a = cu.index('namespace chunk {'); b = cu.index('template<int S,bool F>std::vector<int64_t> ch_resource()', a)
    s = cu[a:b]
    pf = ('  if((k&3)==0){int lo=k?k+8:0,hi=min(32,k+12);if(hi>lo){\n'
          '   if(t<64&&(t&15)==0)' + PF_ASM.format(p='qg+(lo-k)*256', n='(hi-lo)*1024') + '\n'
          '   if(t>=64&&t<96&&((t-64)&7)==0)' + PF_ASM.format(p='sg+(lo-k)*128', n='(hi-lo)*128') + '}}\n')
    s = once(s, '  copy_root+=128;\n\n  if(t<64){', '  copy_root+=128;\n\n' + pf + '  if(t<64){')
    return cu[:a] + s + cu[b:]


def pf_medium(cu):
    a = cu.index('namespace mwbr {'); b = cu.index('}  // namespace mwbr', a)
    s = cu[a:b]
    s = once(s, '   auto issue=[&](int slot){\n',
             '   int pk=0;auto issue=[&](int slot){\n    if((pk&3)==0){int lo=pk?pk+8:0,hi=min(NS,pk+12);if(hi>lo){'
             'if(wq&&(t&15)==0)' + PF_ASM.format(p='qg+(lo-pk)*256', n='(hi-lo)*1024') +
             'if(ws&&((t-64)&7)==0)' + PF_ASM.format(p='sg+(lo-pk)*128', n='(hi-lo)*128') + '}}++pk;\n')
    return cu[:a] + s + cu[b:]


def pf_small(cu):
    # The single producer also prefetches the tile after the one it just issued into L2.
    a = cu.index('__device__ __forceinline__ void prefetch_one('); p, end = balanced(cu, a)
    q = once(p.replace('void prefetch_one(', 'void prefetch_one_pf('), '  if (++st == nstage) { st=0; ph^=1; }',
             '  if (idx < total) ' + PF_ASM.format(p='(pass?w2p:w13p)+(static_cast<size_t>(e)*(pass?NT2:NT1)+m)*TILE_BYTES', n='TILE_BYTES') + '\n  if (++st == nstage) { st=0; ph^=1; }')
    cu = cu[:end] + '\n' + q + cu[end:]
    a = cu.index('template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>'); k, end = balanced(cu, a)
    k2 = once(k, 'prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);',
              'prefetch_one_pf(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);', 2)
    return cu[:a] + k2 + cu[end:]


def xw4(cu):
    # Four warpgroups (4 x 64 output channels) share each 64-token X stage: heavy activation copies from
    # L2 halve, per-thread registers and the per-warpgroup weight stream stay identical; one 512-thread
    # CTA per SM (139 KiB ring at depth 4) replaces two 256-thread CTAs. sm_120 stand-in uses depth 2.
    cu = once(cu, 'constexpr int KT=128,QB=4096,SB=512,WG=2;', 'constexpr int KT=128,QB=4096,SB=512,WG=4;')
    cu = once(cu, 'template<int DEPTH,bool F>__global__ void __launch_bounds__(256,2) kernel(', 'template<int DEPTH,bool F>__global__ void __launch_bounds__(512,1) kernel(')
    cu = once(cu, ' constexpr int S=BR_HOPPER?DEPTH:3,MT=F?64:32;', ' constexpr int S=BR_HOPPER?DEPTH:2,MT=F?32:16;')
    cu = once(cu, 'mt*2+(threadIdx.x>>7),nullptr,nullptr,0x70007000u,red)', 'mt*4+(threadIdx.x>>7),nullptr,nullptr,0x70007000u,red)')
    cu = once(cu, 'int bytes=major==9?chunk::dyn<S,64>():chunk::dyn<3,64>();', 'int bytes=major==9?chunk::dyn<S,64>():chunk::dyn<2,64>();')
    cu = once(cu, '(depth==3?chunk::dyn<3,64>():chunk::dyn<4,64>()):chunk::dyn<3,64>();', '(depth==3?chunk::dyn<3,64>():chunk::dyn<4,64>()):chunk::dyn<2,64>();')
    cu = once(cu, 'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,chunk::kernel<S,F>,256,bytes)', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,chunk::kernel<S,F>,512,bytes)')
    cu = once(cu, 'chunk::kernel<S,F><<<grid,256,bytes,(cudaStream_t)stream>>>', 'chunk::kernel<S,F><<<grid,512,bytes,(cudaStream_t)stream>>>')
    return cu


def resh1(cu):
    # One heavy CTA per SM on Hopper: the depth-4 ring is reserved at half the SM's shared memory, so a
    # small (94 KiB) or up to three medium CTAs co-reside with each heavy CTA and stream while it computes.
    pad = 'major==9?std::max(chunk::dyn<S,64>(),116736):chunk::dyn<3,64>()'
    cu = once(cu, 'int bytes=major==9?chunk::dyn<S,64>():chunk::dyn<3,64>();cudaFuncAttributes a{};', 'int bytes=' + pad + ';cudaFuncAttributes a{};')
    cu = once(cu, 'int bytes=major==9?(depth==3?chunk::dyn<3,64>():chunk::dyn<4,64>()):chunk::dyn<3,64>();',
              'int bytes=major==9?std::max(depth==3?chunk::dyn<3,64>():chunk::dyn<4,64>(),116736):chunk::dyn<3,64>();')
    return cu


def co(cu):
    # Co-schedule an expert's 64-row chunks: tasks are (expert, channel tile, chunk) with the chunk fastest,
    # so adjacent CTAs read the same weight tile together and the reread hits L2 instead of HBM.
    old = (' for(int u=blockIdx.x;u<32*MT;u+=gridDim.x){int e=ORDER[u/MT],mt=u%MT,n=C[e];if(n<=16)continue;\n'
           '  for(int row=0;row<n;row+=64){int rem=min(n-row,64);\n')
    new = (' __shared__ int chunks[33];\n if(threadIdx.x==0){int s=0;chunks[0]=0;for(int i=0;i<32;i++){int n=C[ORDER[i]];s+=n>16?(n+63)/64:0;chunks[i+1]=s;}}\n __syncthreads();\n'
           ' for(int u=blockIdx.x;u<chunks[32]*MT;u+=gridDim.x){int ix=0;\n  for(int d=16;d;d>>=1)if(chunks[ix+d]*MT<=u)ix+=d;\n'
           '  int nc=chunks[ix+1]-chunks[ix],rel=u-chunks[ix]*MT,mt=rel/nc,e=ORDER[ix],n=C[e];\n'
           '  {int row=(rel%nc)*64,rem=min(n-row,64);\n')
    return once(cu, old, new)


ARMS = {
    'PF': dict(mechanism='L0N - PFN / weight bytes in flight: in-kernel L2 bulk prefetch of each branch weight stream ahead of its ring',
               P3B_CUDA=lambda cu: pf_small(pf_medium(pf_heavy(cu))), MT_PMZ_CUDA=pf_small),
    'RESH1': dict(mechanism='RES1/RES3 + HISTN branch serialization: one heavy CTA per SM so small/medium CTAs stream beside heavy compute',
                  P3B_CUDA=resh1),
    'CO': dict(mechanism='L0N - REUSE / weight rereads: an expert\'s 64-row chunks run on adjacent CTAs so rereads share L2', P3B_CUDA=co),
    'XW4': dict(mechanism='L0N - NOX / activation copies: four warpgroups share each heavy X stage (half the heavy L2 activation traffic), one 512-thread CTA/SM',
                P3B_CUDA=xw4),
}


def literal_assign(source, name):
    t = ast.parse(source)
    node = next(n for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == name)
    lines = source.splitlines(True)
    start = sum(len(l) for l in lines[:node.lineno - 1]) + node.col_offset
    end = sum(len(l) for l in lines[:node.end_lineno - 1]) + node.end_col_offset
    return start, end, ast.literal_eval(node.value)


def candidate(source, spec):
    text = source
    for name in ('P3B_CUDA', 'MT_PMZ_CUDA'):
        if name in spec:
            a, b, value = literal_assign(text, name)
            new = spec[name](value)
            assert new != value
            text = text[:a] + name + '=' + repr(new) + text[b:]
    compile(text, '<p3b-split-candidate>', 'exec')
    return text


def main(names=None):
    out = H / 'out'; out.mkdir(exist_ok=True)
    source = s22_source(); base_lines = source.splitlines(True); specs = {}
    for k, name in enumerate(names or ARMS):
        text = candidate(source, ARMS[name])
        (out / f'prod_{name}.py').write_text(text)
        shipped = '\n'.join(line.rstrip() for line in pack(text).splitlines()) + '\n'
        (H / f'{name}.py').write_text(shipped)
        edits = [(a, b, ''.join(text.splitlines(True)[c:d])) for op, a, b, c, d in
                 difflib.SequenceMatcher(None, base_lines, text.splitlines(True), autojunk=False).get_opcodes() if op != 'equal']
        specs[name] = dict(sha256=sha(text), shipped_sha256=sha(shipped), edits=edits, rows=ROWS, mechanism=ARMS[name]['mechanism'], kind=k)
        for unit in ('P3B_CUDA', 'MT_PMZ_CUDA'):
            (out / f'prod_{name}_{unit}.cu').write_text(literal_assign(text, unit)[2])
    (out / 'base.py').write_text(source)
    (out / 'arms.json').write_text(json.dumps(specs))
    (H / 'candidates.json').write_text(json.dumps(dict(base_sha256=sha(source), base_file='problems/muse_decode_layer/cand_r3_best7_s22_final.py',
        arms={k: {a: b for a, b in v.items() if a != 'edits'} for k, v in specs.items()}), indent=2) + '\n')
    print('built', list(specs), 'on S22', sha(source)[:12])


if __name__ == '__main__':
    main(sys.argv[1:] or None)
