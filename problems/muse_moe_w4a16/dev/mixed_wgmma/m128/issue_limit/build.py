"""Freeze exact instruction-cut variants and a phase-instrumented NT1 control."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent; M128=HERE.parent; BASE=M128/'scale_bank'; ROOT=HERE.parents[5]
sys.path[:0]=[str(BASE),str(M128),str(M128.parent),str(ROOT/'tools')]
from source import source
from build_repair_lab import rename,pointer_bindings
from native_abi import PRELUDE,CPP_PRELUDE,lean
from build_prefill_lab import subset
from okc_run import rejected_words
import importlib.util
spec=importlib.util.spec_from_file_location('scale_builder',BASE/'build.py')
scale_builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(scale_builder)


def literal(path,name):
    tree=ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(getattr(t,'id','')==name for t in n.targets))


def blend(s):
    old='unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;'
    new='unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));'
    assert old in s
    return s.replace(old,new)


def word_scales(s):
    # Store an integer exponent/mantissa word per original E4M3 block. This is
    # reversible metadata, not a prepared dequantized expert-weight matrix.
    s=blend(s)
    s=s.replace('64*KT/16','64*KT/8')
    s=s.replace('unsigned b,unsigned raw,unsigned*out)', 'unsigned b,unsigned long long raw,unsigned*out)')
    for i,selector in enumerate(('0x1010u','0x3232u','0x1010u','0x3232u')):
        old=f'unsigned s{i}=__byte_perm(raw,0u,0x{(0x4040+0x101*i):04x}u)*16u+0x70007000u;'
        value='(unsigned)raw' if i<2 else '(unsigned)(raw>>32)'
        assert old in s,old
        s=s.replace(old,f'unsigned s{i}=__byte_perm({value},0u,{selector});')
    old='async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);'
    assert old in s
    s=s.replace(old,'if(l<16)async_copy(q+(g*128+w*32)*8+l*16,Q+z*64+l*4,16,1);')
    old='if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);'
    assert old in s
    s=s.replace(old,'if(l<8)async_copy(s+g*256+w*64+l*8,S+z*64+l*8,8,1);')
    old='unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);'
    assert old in s
    s=s.replace(old,'unsigned long long sf=*(const unsigned long long*)(ss+st*SB+z*256+w*64+(l>>2)*8);')
    return s


def nt1(prefix,lop=False,trace=False):
    s=literal(ROOT/'problems/muse_decode_layer/cand_r3_v9.py','V9_CUDA_SRC')
    s=lean(s[:s.index('int64_t max_smem_attr()')]).replace('namespace {','namespace '+prefix+' {',1)
    if lop:
        old='const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);'
        assert old in s
        s=s.replace(old,'uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));')
    if trace:
        s=s.replace('int M, int local_start, int nstage) {','int M, int local_start, int nstage,unsigned long long* trace) {',1)
        a=s.index('template <int NT>')
        s=s[:a]+'''__device__ __forceinline__ unsigned long long stamp(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
'''+s[a:]
        s=s.replace('  constexpr int NB = 8 * NT;','  __shared__ unsigned long long ticks[6];if(threadIdx.x==0)ticks[0]=stamp();\n  constexpr int NB = 8 * NT;',1)
        s=s.replace('  for (int pass = 0; pass < 2; ++pass) {','  if(tid==0)ticks[1]=stamp();\n  for (int pass = 0; pass < 2; ++pass) {\n    if(pass==1&&tid==0)ticks[2]=stamp();',1)
        s=s.replace('  // ---------------- grid barrier ----------------','  if(tid==0)ticks[3]=stamp();\n  // ---------------- grid barrier ----------------',1)
        s=s.replace('  // ---------------- combine: item','  if(tid==0)ticks[4]=stamp();\n  // ---------------- combine: item',1)
        s=s.replace('\n}\n\nsize_t smem_bytes(', '\n  if(tid==0){ticks[5]=stamp();for(int i=0;i<5;++i)trace[blockIdx.x*5+i]=ticks[i+1]-ticks[i];}\n}\n\nsize_t smem_bytes(',1)
    extra=',(unsigned long long*)v[13]' if trace else ''
    s+=f'''
int64_t {prefix}_init(){{int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));cudaFuncAttributes attr{{}};WX_CUDA_CHECK(cudaFuncGetAttributes(&attr,{prefix}::moe_fused<1>));opt-=attr.sharedSizeBytes;WX_CUDA_CHECK(cudaFuncSetAttribute({prefix}::moe_fused<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt));return opt;}}
int64_t {prefix}_smem(int64_t stages,int64_t M){{return {prefix}::smem_bytes(1,stages,M);}}
int64_t {prefix}_resident(int64_t stages,int64_t M){{int n;WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,{prefix}::moe_fused<1>,{prefix}::NTHR,{prefix}::smem_bytes(1,stages,M)));return n;}}
void {prefix}_run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream){{
 {prefix}::moe_fused<1><<<grid,{prefix}::NTHR,{prefix}::smem_bytes(1,stages,M),(cudaStream_t)stream>>>((__nv_bfloat16*)v[0],(int*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(uint8_t*)v[4],(uint8_t*)v[5],(float*)v[6],(float*)v[7],(__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages{extra});WX_CUDA_CHECK(cudaGetLastError());}}
'''
    return PRELUDE+s


def main():
    raw=source(); rec=scale_builder.recode_source(raw)
    codes={p:pointer_bindings(rename(body,'rot64',(3,))).replace('rot64',p)
           for p,body in (('ibase',rec),('iword',word_scales(rec)))}
    dist=(M128.parent/'w2fix.cu').read_text()
    dist=word_scales(scale_builder.recode_source(dist))
    dist=dist.replace('namespace mwfx','namespace idist').replace('mwfx::','idist::')
    dist=re.sub(r'\bfx_','idist_',dist)
    codes['idist']=dist
    for p,lo,tr in (('norig',False,False),('nlop',True,False),('ntrace',False,True)):
        codes[p]=nt1(p,lo,tr)
    funs=[];cpp=CPP_PRELUDE
    for p in ('ibase','iword'):
        funs += [p+'_'+f for f in ('init','info','route','run')]
        cpp+=f'int64_t {p}_init();std::vector<int64_t> {p}_info(int64_t,int64_t);void {p}_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void {p}_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
    cpp+='int64_t idist_init();std::vector<int64_t> idist_info(int64_t,int64_t,int64_t);void idist_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);\n'
    funs += ['idist_'+f for f in ('init','info','route','run','finish')]
    for p in ('norig','nlop','ntrace'):
        cpp+=f'int64_t {p}_init();int64_t {p}_smem(int64_t,int64_t);int64_t {p}_resident(int64_t,int64_t);void {p}_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
        funs += [p+'_'+f for f in ('init','smem','resident','run')]
    old=(ROOT/'prof/lab/moe_m128_scale_bank.py').read_text()
    helpers=subset(old,('dc_weights','dc_fixtures','dc_oracles','MWGraph','mw_repack_planes','_encode_scales'))
    oldtree=ast.parse(old)
    prep=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='scale_prepare')
    prep.name='issue_prepare';prep.args.args=prep.args.args[:1]
    # Keep only the recoded comparison arm and remove the old module binding.
    class Fold(ast.NodeTransformer):
        def visit_If(self,n):
            n=self.generic_visit(n)
            if isinstance(n.test,ast.Name) and n.test.id=='recoded':return n.body
            return n
    prep=Fold().visit(prep)
    prep.body=[n for n in prep.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and ast.unparse(t)=="bank['mod']" for t in n.targets))]
    ntpack=subset((ROOT/'problems/muse_decode_layer/cand_r3_v9.py').read_text(),('v9_prepare',))
    tree=ast.parse(ntpack)
    tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom,ast.Try))]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name=='v9_prepare':
            n.name='issue_nt_prepare'
            for node in ast.walk(n):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_v9_module':node.func.id='issue_module'
    # _v9_module is unreachable after the replacement; no Torch-header build is needed.
    tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='_v9_module']
    tree.body=[n for n in tree.body if not isinstance(n,ast.Assign) or not any(getattr(t,'id','') in ('V9_CUDA_SRC','V9_CPP_SRC','_V9_MOD') for t in n.targets)]
    cuda='\n'.join(codes.values())
    for p,s in codes.items():(HERE/(p+'.cu')).write_text(s)
    text=helpers+'\n'+ast.unparse(prep)+'\n'+ast.unparse(tree)+'\nimport functools\nimport gc\nfrom torch.utils.cpp_extension import load_inline\n'
    text+='ISSUE_CUDA='+repr(cuda)+'\nISSUE_CPP='+repr(cpp)+'\nISSUE_FUNCTIONS='+repr(funs)+'\n'+(HERE/'driver.py.inc').read_text()
    assert not rejected_words(text),rejected_words(text)
    out=ROOT/'prof/lab/moe_m128_issue_limit.py';out.write_text(text)
    manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),
        sources={p:hashlib.sha256(s.encode()).hexdigest() for p,s in codes.items()},
        base_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),nt1_source='problems/muse_decode_layer/cand_r3_v9.py',
        seeds=[5353+7919*i for i in range(12)],fixtures_per_seed=3,rounds=6,cold_bytes=128<<20,
        encoding='u16 integer word = 0x7000+(u8 recode<<4); reversible E4M3 metadata; packed E2M1 unchanged')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out.relative_to(ROOT),manifest['sha256'],len(text))


if __name__=='__main__':main()
