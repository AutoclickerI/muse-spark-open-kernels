"""Split S19 activation transport, cached-register lifetime, MMA recurrence and padding.

Oracle outputs are ignored. Only BASE, BASE_PAD, PAD_CTRL and LAZY are TASK-bearing.
No donor code is used; frozen S19 and the prior ladder provide all math and launch glue.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[5]
O = R / 'runs/screen/p3a_operand'
O.mkdir(parents=True, exist_ok=True)

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

ip = load('operand_ip', D.parent / 'intpipe/build.py')
lad = load('operand_ladder', D.parent / 'ladder/build.py')
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
MODES = {'FRAG_SMEM': 1, 'FRAG_REG': 2, 'FRAG_JIT': 3, 'ACC_FREE': 4, 'ACC_CTRL': 5,
         'PAD_USEFUL': 6, 'PAD_CTRL': 7, 'LAZY': 9, 'BASE_PAD': 10}

def operand(cu):
    a = cu.index('template <int NT, int F>\n__global__')
    brace = cu.index('{', a)
    b, depth = brace + 1, 1
    while depth:
        depth += (cu[b] == '{') - (cu[b] == '}')
        b += 1
    k = cu[a:b].replace('template <int NT, int F>', 'template <int NT,int F,int K>').replace(' moe_var(', ' op_var(')
    needle = '  const int q = lane & 3;'
    assert k.count(needle) == 1
    k = k.replace(needle, '''  // The synthetic bank is private, never aliases routing/reduction scratch.
  // One register-order fragment pattern is repeated across K-groups: pack-free timing oracle only.
  uint32_t* frag = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(pair_of+npairs)+127ull)&~127ull);
  if constexpr(K==1||K==3){for(int z=tid;z<2048;z+=NTHR)frag[z]=0x3f003f00u;bar_consumers();}
''' + needle)
    start = k.index('#pragma unroll\n', k.index('probe[16] += tg() - wait0;'))
    end = k.index('    }\n    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp)', start)
    original_fill = k[start:end]
    fill = '''      if constexpr(K==1){
        // Preserve all 64 cached Bf registers, replace global gathers with volatile shared v4.
#pragma unroll
        for(int j=0;j<NT;++j)for(int g=0;g<8;++g){
          uint4 v0=op_ld4(frag+tid*4),v1=op_ld4(frag+1024+tid*4);
          Bf[j][g][0]=v0.x;Bf[j][g][1]=v0.y;Bf[j][g][2]=v0.z;Bf[j][g][3]=v0.w;
          Bf[j][g][4]=v1.x;Bf[j][g][5]=v1.y;Bf[j][g][6]=v1.z;Bf[j][g][7]=v1.w;
        }
      }else if constexpr(K==2){
#pragma unroll
        for(int j=0;j<NT;++j)for(int g=0;g<8;++g)for(int p=0;p<8;++p)Bf[j][g][p]=0x3f003f00u;
      }else if constexpr(K!=3&&K!=9){
''' + original_fill + '      }\n'
    k = k[:start] + fill + k[end:]
    # For the correct LAZY arm use the identical original loads, but load one g at each weight tile.
    # These arrays last for a single group, not an expert/token-block segment.
    group = '''        uint32_t Bj[NT][8];
        if constexpr(K==3){
#pragma unroll
          for(int j=0;j<NT;++j){uint4 u=op_ld4(frag+tid*4),v=op_ld4(frag+1024+tid*4);
            Bj[j][0]=u.x;Bj[j][1]=u.y;Bj[j][2]=u.z;Bj[j][3]=u.w;
            Bj[j][4]=v.x;Bj[j][5]=v.y;Bj[j][6]=v.z;Bj[j][7]=v.w;}
        }else if constexpr(K==9){
#pragma unroll
          for(int j=0;j<NT;++j){int idx=nb*NB+j*8+(lane>>2);uint4 u=make_uint4(0,0,0,0),v=u;
            if(idx<ne){int pos=off+idx;const __nv_bfloat16* row=pass?inter+(size_t)pos*HD:x+(size_t)(pair_of[pos]>>3)*HD;
              const uint4* p=reinterpret_cast<const uint4*>(row+warp*512+g*64+q*16);
              if(pass){u=__ldcg(p);v=__ldcg(p+1);}else{u=__ldg(p);v=__ldg(p+1);}}
            Bj[j][0]=u.x;Bj[j][1]=u.y;Bj[j][2]=u.z;Bj[j][3]=u.w;
            Bj[j][4]=v.x;Bj[j][5]=v.y;Bj[j][6]=v.z;Bj[j][7]=v.w;}
        }
'''
    needle = '        const uint4 wv = wq[g * 32];'
    assert k.count(needle) == 1
    k = k.replace(needle, group + needle)
    needle = '          for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);'
    assert k.count(needle) == 1
    k = k.replace(needle, '''          for(int j=0;j<NT;++j){
            // Paired volatile sinks retain ALL decode work when a warp's MMA is omitted.
            if constexpr(K==6||K==7)op_sink(red+buf*NCW*32*RED,tid,a0,a1,a2,a3);
            if constexpr(K==4){acc[j][0]=acc[j][1]=acc[j][2]=acc[j][3]=0.f;}
            // Deliberately wrong math: omit K-warps to realize the ideal useful-N instruction budget.
            // This counts instructions; it is not an implementation of narrower BF16 tensor MMA.
            if(K!=6||warp<min(8,max(0,ne-nb*NB)))
              mma16816(acc[j],a0,a1,a2,a3,(K==3||K==9)?Bj[j][2*s]:Bf[j][g][2*s],(K==3||K==9)?Bj[j][2*s+1]:Bf[j][g][2*s+1]);
            if constexpr(K==4||K==5)op_sink(red+buf*NCW*32*RED,tid,__float_as_uint(acc[j][0]),__float_as_uint(acc[j][1]),__float_as_uint(acc[j][2]),__float_as_uint(acc[j][3]));
          }''')
    helper = '''
__device__ __forceinline__ uint4 op_ld4(const uint32_t* p){uint4 v;
  asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(smem_u32(p)):"memory");return v;}
__device__ __forceinline__ void op_sink(float* p,int t,uint32_t a,uint32_t b,uint32_t c,uint32_t d){
  asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(smem_u32(p)+t*16),"r"(a),"r"(b),"r"(c),"r"(d):"memory");}
'''
    cu = cu[:b] + helper + k + cu[b:]
    suffix = '''
template<int F,int K>void op_launch(std::vector<int64_t>&v,int M,int start,int stages,int grid,cudaStream_t st){
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=smem_bytes(1,stages,M)+((K==1||K==3||K==10)?8320:0);cfg.stream=st;
  cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&a;cfg.numAttrs=1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,op_var<1,F,K>,(const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],(__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages,(uint64_t*)v[13]));}
void op_run(std::vector<int64_t>v,int M,int start,int stages,int grid,int64_t stream,int F,int mode){
#define GO(K) if(mode==K){if(F==95)op_launch<95,K>(v,M,start,stages,grid,(cudaStream_t)stream);else op_launch<69,K>(v,M,start,stages,grid,(cudaStream_t)stream);return;}
GO(1) GO(2) GO(3) GO(4) GO(5) GO(6) GO(7) GO(9) GO(10)
#undef GO
throw std::runtime_error("operand mode");}
std::vector<int64_t>op_info(int M,int F,int mode){cudaFuncAttributes a{};int n=0,extra=(mode==1||mode==3||mode==10)?8320:0;
#define INFO(K) case K:{auto k=F==95?op_var<1,95,K>:op_var<1,69,K>;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,256,smem_bytes(1,2,M)+extra));break;}
switch(mode){INFO(1) INFO(2) INFO(3) INFO(4) INFO(5) INFO(6) INFO(7) INFO(9) INFO(10)default:throw std::runtime_error("operand info");}
#undef INFO
return{n,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)smem_bytes(1,2,M)+extra,(int64_t)a.sharedSizeBytes};}
void op_init(){int d,opt;C10_CUDA_CHECK(cudaGetDevice(&d));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));
#define INIT(K,F) {cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,op_var<1,F,K>));C10_CUDA_CHECK(cudaFuncSetAttribute(op_var<1,F,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));}
#define BOTH(K) INIT(K,69) INIT(K,95)
BOTH(1) BOTH(2) BOTH(3) BOTH(4) BOTH(5) BOTH(6) BOTH(7) BOTH(9) BOTH(10)
#undef BOTH
#undef INIT
}
'''
    return cu + suffix

def main():
    original = ip.original()
    cells = {}
    for i, prefix in enumerate(ip.PREFIXES):
        c = ip.cell(original, prefix)
        tree = ast.parse(c)
        cn = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '').endswith('_CUDA'))
        pn = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '').endswith('_CPP'))
        old = ast.literal_eval(cn.value)
        # Retain the prior ladder L2/L3 controls in the exact same TU.
        cu = operand(lad.diagnostic(old))
        cpp = ast.literal_eval(pn.value) + '\nvoid lad_run(std::vector<int64_t>,int,int,int,int,int64_t,int,int,bool);void lad_init();std::vector<int64_t>lad_info(int,int,int,bool);void op_run(std::vector<int64_t>,int,int,int,int,int64_t,int,int);void op_init();std::vector<int64_t>op_info(int,int,int);\n'
        c = c.replace(ast.get_source_segment(c, cn), cn.targets[0].id + '=' + repr(cu), 1)
        c = c.replace(ast.get_source_segment(ip.cell(original, prefix), pn), pn.targets[0].id + '=' + repr(cpp), 1)
        c = c.replace("'epi_info'", "'epi_info','lad_run','lad_init','lad_info','op_run','op_info','op_init'")
        c = c.replace('mod.smem_optin = mod.max_smem_attr()', 'mod.smem_optin = mod.max_smem_attr();mod.lad_init();mod.op_init()')
        tag = 'opa' + str(i)
        c = c.replace(prefix, tag).replace(prefix.upper(), tag.upper())
        p = D / (tag + '_cell.py')
        p.write_text(c)
        (O / (tag + '.cu')).write_text(cu)
        cells[str(i)] = dict(cell=str(p.relative_to(R)), cell_sha256=sha(c), cuda_sha256=sha(cu), prefix=tag, original_cuda_sha256=sha(old))
    (D / 'cells.json').write_text(json.dumps(cells, indent=2) + '\n')
    (D / 'modes.json').write_text(json.dumps(MODES, indent=2) + '\n')
    print('OPERAND_BUILT', list(cells), MODES)

if __name__ == '__main__':
    main()
