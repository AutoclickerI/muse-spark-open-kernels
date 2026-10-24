"""Generate the one-launch streaming projection + boundary lab."""
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent
STREAM=HERE.parent/'streaming'
src=(STREAM/'kernel.cu').read_text()
pos=src.index('// No spinning, no extra launch:')
src=src[:pos]+'} // namespace stream_proj\n'+(HERE/'glue.cuh').read_text()+'\nnamespace stream_proj {\n'+src[pos:]
# A split tile returns whether this CTA owns the full FP32 result.
src=src.replace('__device__ __forceinline__ void finish(', '__device__ __forceinline__ bool finish(')
src=src.replace('  }\n}\n\n// A single lane-preserving', '  }\n  return last;\n}\n\n// A single lane-preserving')
src=src.replace('int M, int N, int K, int splits, bool fp32) {',
                'int M, int N, int K, int splits, bool fp32, Glue gl) {')
src=src.replace('  float a[ROWS][8] = {};',
                '  __shared__ float input_inv[1];\n  if(gl.prologue) init_input<1,WARPS*32>(gl,x,M,K,input_inv);\n  float a[ROWS][8] = {};')
src=src.replace('*reinterpret_cast<const uint4*>(x + (size_t)tok * K + k)',
                'input16(gl,x,tok,k,K,gl.prologue?input_inv[0]:1.f)')
src=src.replace('  if (splits > 1) finish<R, 1, WARPS * 32>(partial, counters, y, M, N, splits, nt, M, fp32);',
                '  bool complete=splits==1 || finish<R,1,WARPS*32>(partial,counters,y,M,N,splits,nt,M,fp32);\n  if(complete) publish<R,1,WARPS*32>(gl,(float*)y,M,N,nt);')
src=src.replace('  auto load = [&](int tile) {',
                '  __shared__ float input_inv[T];\n  if(gl.prologue) init_input<T,WARPS*32>(gl,x,M,K,input_inv);\n  auto load = [&](int tile) {')
src=src.replace('      cp16(s + r * STRIDE + c, src, valid);',
                '''      if(!is_w && gl.prologue) {
        uint4 v=valid?input16(gl,x,global_r,kb+c,K,input_inv[r-R]):uint4{};
        *(uint4*)(s+r*STRIDE+c)=v;
      } else cp16(s + r * STRIDE + c, src, valid);''')
src=src.replace('  if (splits > 1) finish<R, T, WARPS * 32>(partial, counters, y, M, N, splits, nt, mt, fp32);',
                '  bool complete=splits==1 || finish<R,T,WARPS*32>(partial,counters,y,M,N,splits,nt,mt,fp32);\n  if(complete) publish<R,T,WARPS*32>(gl,(float*)y,M,N,nt);')
src=src.replace('bool fp32, cudaStream_t stream) {','bool fp32, Glue gl, cudaStream_t stream) {')
src=src.replace('(w,x,y,p,c,M,N,K,splits,fp32);','(w,x,y,p,c,M,N,K,splits,fp32,gl);')
# Both consumers in one library, with the vector consumer selected by config+100.
start=src.index('template<int R, int T, int BK, int STAGES, int WARPS>')
end=src.index('template<int U, int W, int R>\nint launch_scalar',start)
vector=src[start:end].replace('void tensor(', 'void tensor_vector(').replace('BK + 8','BK + 32')
a=vector.index('#pragma unroll\n    for (int k = 0; k < BK; k += 16)')
b=vector.index('    __syncthreads();',a)
vector=vector[:a]+(STREAM/'consumer_vector.cu.inc').read_text()+vector[b:]
src=src[:end]+vector+src[end:]
start=src.index('template<int R, int T, int BK, int S, int W>')
end=src.index('} // namespace stream_proj',start)
launch=src[start:end].replace('launch_tensor(', 'launch_tensor_vector(').replace('BK + 8','BK + 32').replace('tensor<R,T,BK,stages,W>', 'tensor_vector<R,T,BK,stages,W>')
src=src[:end]+launch+src[end:]
src=src.replace('int splits, int fp32, void* stream) {','int splits, int fp32, const void* glue, void* stream) {')
src=src.replace('  if (K % (splits * 8))', '  Glue gl=glue?*(const Glue*)glue:Glue{};\n  if (K % (splits * 8))')
src=src.replace('K,splits,(bool)fp32,(cudaStream_t)stream','K,splits,(bool)fp32,gl,(cudaStream_t)stream')
src=src.replace('__launch_bounds__(WARPS * 32)','__launch_bounds__(WARPS * 32,1)')
start=src.index('  switch(config) {')
end=src.index('    default:',start)
vector_cases='\n'.join(l.replace('case '+l.split('case ')[1].split(':')[0]+':', 'case '+str(int(l.split('case ')[1].split(':')[0])+100)+':').replace('launch_tensor<','launch_tensor_vector<')
                       for l in src[start:end].splitlines() if 'return launch_tensor<' in l)+'\n'
src=src[:end]+vector_cases+src[end:]
out=HERE.parent/'streaming_glue_lab.py'
helpers=(STREAM/'lab.py.inc').read_text()
helpers='\n\n'.join(ast.get_source_segment(helpers,n) for n in ast.parse(helpers).body
                       if isinstance(n,(ast.Import,ast.ImportFrom)) or
                       isinstance(n,ast.FunctionDef) and n.name in ('sp_inputs','sp_check'))
out.write_text('"""Generated streaming projections with in-launch boundary continuations."""\nSP_CUDA = '+repr(src)+'\n\n'
               +(STREAM/'runtime.py.inc').read_text().replace('[_sp_ct.c_int] * 6 + [_sp_ct.c_void_p]','[_sp_ct.c_int] * 6 + [_sp_ct.c_void_p] * 2')
                .replace('            torch.cuda.current_stream(x.device).cuda_stream,','            None, torch.cuda.current_stream(x.device).cuda_stream,')
               +'\n\n'+helpers
               +'\n\n'+(HERE/'runtime.py.inc').read_text()+'\n\n'+(HERE/'lab.py.inc').read_text())
print(out)
