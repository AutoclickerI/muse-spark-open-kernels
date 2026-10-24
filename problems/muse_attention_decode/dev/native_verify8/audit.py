"""Compare complete 64-key native-WGMMA and widened-FP16 tiles in sm90a SASS."""
import collections,hashlib,json,re,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def probe(source,native):
    if native:
        state='WGState st;'
        init=''
        call='native_tile<false,false>(st,sm,sm+4096,sm+8192,sm+12288,sm+16384,0,64,80000,scale,trace);'
    else:
        state='State st;'
        init='''
#pragma unroll
 for(int k=0;k<4;k++)for(int j=0;j<4;j++)st.q[k][j]=q[threadIdx.x*16+4*k+j];
'''
        call='widen_stage<64>(sm,sm+4096,sm+8192,sm+16384);__syncthreads();tile<64,false,64,false>(st,sm+8192,sm+16384,0,64,80000,scale,0,trace);'
        source=source.replace(' for(int i=tid;i<N*32;i+=128){','#pragma unroll\n for(int step=0;step<N/4;step++){int i=tid+step*128;')
    source+='''
__global__ void tile_probe(const float* initial,const uint32_t* q,float* out,float scale){
 extern __shared__ __align__(1024) uint8_t sm[];
 STATE Trace trace{};
 INIT
#pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)st.o[n][e]=initial[threadIdx.x*36+4*n+e];
 st.m[0]=initial[threadIdx.x*36+32];st.m[1]=initial[threadIdx.x*36+33];
 st.l[0]=initial[threadIdx.x*36+34];st.l[1]=initial[threadIdx.x*36+35];
 CALL
#pragma unroll
 for(int n=0;n<8;n++)for(int e=0;e<4;e++)out[threadIdx.x*36+4*n+e]=st.o[n][e];
 out[threadIdx.x*36+32]=st.m[0];out[threadIdx.x*36+33]=st.m[1];
 out[threadIdx.x*36+34]=st.l[0];out[threadIdx.x*36+35]=st.l[1];
}
'''.replace('STATE',state).replace('INIT',init).replace('CALL',call)
    return source


def entries(sass):
    result=[]
    for part in sass.split('Function : ')[1:]:
        symbol=part.splitlines()[0]
        ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]+)',part);counts=collections.Counter(ops)
        result.append(dict(symbol=symbol,instructions=len(ops),per_key=len(ops)/64,opcodes=dict(counts)))
    return result


def main():
    rows=[]
    for name,path in [('native',HERE/'native.cu'),('f16',ROOT/'problems/muse_attention_decode/dev/producer_phases/widened.cu')]:
        src=probe(path.read_text(),name=='native');file=HERE/(name+'_probe.cu');file.write_text(src)
        with tempfile.TemporaryDirectory() as tmp:
            cubin=Path(tmp)/'kernel.cubin'
            cmd=['nvcc','-O3','-std=c++17','-arch=sm_90a','-lineinfo','--cubin','--ptxas-options=-v,--warn-on-spills',str(file),'-o',str(cubin)]
            p=subprocess.run(cmd,capture_output=True,text=True,check=True)
            sass=subprocess.run(['cuobjdump','--dump-sass',str(cubin)],capture_output=True,text=True,check=True).stdout
            resource=subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],capture_output=True,text=True,check=True).stdout
            kernels=entries(sass);tile=next(r for r in kernels if 'tile_probe' in r['symbol'])
            tile_sass=next(p for p in sass.split('Function : ') if p.startswith(tile['symbol']))
            # Keep instruction-count evidence and small exact machine-code excerpts;
            # the complete dump is reproduced by this script.
            tile_lines=tile_sass.splitlines();chosen=set()
            hits=[i for i,line in enumerate(tile_lines) if any(op in line for op in ('QGMMA.','HGMMA.'))]
            if not hits:
                hits=[i for i,line in enumerate(tile_lines) if 'HMMA.' in line][:2]
                hits += [i for i,line in enumerate(tile_lines) if 'F2FP.F16.E4M3' in line][:2]
            for i in hits:
                chosen.update(range(max(0,i-4),min(len(tile_lines),i+5)))
            excerpt='Function : '+tile['symbol']+'\n'+'\n'.join(tile_lines[i] for i in sorted(chosen))+'\n'
            (HERE/(name+'_tile.sass')).write_text(excerpt)
            rows.append(dict(name=name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(src.encode()).hexdigest(),ptxas=p.stdout+p.stderr,resources=resource,kernels=kernels,tile=tile))
            print(name,json.dumps(tile,sort_keys=True),flush=True)
    native,f16=rows
    assert native['tile']['instructions']<f16['tile']['instructions'],rows
    assert any(k.startswith(('HGMMA','QGMMA')) and 'E4M3' in k for k in native['tile']['opcodes'])
    assert not any(k.startswith('HMMA') for k in native['tile']['opcodes'])
    report=dict(architecture='sm90a',tile_keys=64,tile_query_rows=64,selected='native WGMMA',
        scope='static instructions of complete interior tile plus state load/store; widening loop fully unrolled. Includes transpose/softmax/P packing and MMA. Not a dynamic issue or timing measurement.',rows=rows)
    (HERE/'sass_comparison.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
