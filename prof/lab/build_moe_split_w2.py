"""Derive K-partitioned W2 from the balanced native V9 tile pipeline."""
import ast
from pathlib import Path
from build_moe_v9_reuse import literal,ROOT


def build():
    p=ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/balanced.py';s=p.read_text();cu=literal(s,'BAL_CUDA')
    a=cu.index('template<int NT>\n__global__');b=cu.index('\nsize_t balance_smem',a)
    k=cu[a:b].replace('template<int NT>','template<int NT,int SK>').replace('balance_gemm_k(','split_gemm_k(')
    k=k.replace('__launch_bounds__(NTHR, NT==1?2:1)','__launch_bounds__((SK<8?8/SK:1)*32, 1)')
    k=k.replace('int pass,int chunk,int nstage){','int pass,int chunk,int nstage,int P){')
    k=k.replace(' constexpr int NB=8*NT,RED=4*NT;',
''' pass=1;
 constexpr int NWP=SK<8?8/SK:1,NTP=NWP*32,NG=SK==16?4:8;
 constexpr int PART_W_BYTES=32768/SK,PART_S_BYTES=SK==16?512:4096/SK;
 constexpr int PART_BYTES=PART_W_BYTES+PART_S_BYTES,NB=8*NT,RED=4*NT;''')
    k=k.replace('nstage*TILE_BYTES','nstage*PART_BYTES').replace('NCW','NWP').replace('NTHR','NTP')
    k=k.replace('int total=plan[pass?65:32]','int total=plan[65]*SK')
    k=k.replace('job->task=atomicAdd(plan+68+pass,1);if(job->task<total){','job->task=atomicAdd(plan+69,1);if(job->task<total){int task=job->task/SK;job->split=job->task%SK;')
    k=k.replace('<=job->task','<=task').replace('rel=job->task-pref[a]','rel=task-pref[a]')
    k=k.replace('const int e=job->e,off=job->off,ne=job->ne,m0=job->m0,m1=job->m1;',
'''const int e=job->e,off=job->off,ne=job->ne,m0=job->m0,m1=job->m1,split=job->split;
  const int basew=SK==16?split/2:split*NWP,gbase=SK==16?(split&1)*4:0;
  const int sfOffset=SK==16?(split/2)*512:split*PART_S_BYTES;''')
    k=k.replace('Bf[NT][8][8]','Bf[NT][NG][8]').replace('g<8','g<NG')
    k=k.replace('warp*512+g*64+q*16','(basew+warp)*512+(gbase+g)*64+q*16')
    k=k.replace('warp*8*32+lane','warp*NG*32+lane').replace('tile+TILE_W_BYTES','tile+PART_W_BYTES').replace('scw[g>>1]','scw[(g+gbase)>>1]')
    old='''if(tid==0){for(int s=0;s<nstage&&m0+s<m1;++s){mbar_expect_tx(bars+s,TILE_BYTES);bulk_g2s(tiles+s*TILE_BYTES,bank+((size_t)e*ntile+m0+s)*TILE_BYTES,TILE_BYTES,bars+s,pol);}}'''
    new='''if(tid==0){for(int s=0;s<nstage&&m0+s<m1;++s){const uint8_t* src=bank+((size_t)e*ntile+m0+s)*TILE_BYTES;
   mbar_expect_tx(bars+s,PART_BYTES);bulk_g2s(tiles+s*PART_BYTES,src+split*PART_W_BYTES,PART_W_BYTES,bars+s,pol);
   bulk_g2s(tiles+s*PART_BYTES+PART_W_BYTES,src+32768+sfOffset,PART_S_BYTES,bars+s,pol);}}'''
    assert old in k;k=k.replace(old,new)
    old='''if(tid==0&&m+nstage<m1){mbar_expect_tx(bars+st,TILE_BYTES);bulk_g2s(tiles+st*TILE_BYTES,bank+((size_t)e*ntile+m+nstage)*TILE_BYTES,TILE_BYTES,bars+st,pol);}'''
    new='''if(tid==0&&m+nstage<m1){const uint8_t* src=bank+((size_t)e*ntile+m+nstage)*TILE_BYTES;
    mbar_expect_tx(bars+st,PART_BYTES);bulk_g2s(tiles+st*PART_BYTES,src+split*PART_W_BYTES,PART_W_BYTES,bars+st,pol);
    bulk_g2s(tiles+st*PART_BYTES+PART_W_BYTES,src+32768+sfOffset,PART_S_BYTES,bars+st,pol);}'''
    assert old in k;k=k.replace(old,new).replace('tile=tiles+st*TILE_BYTES','tile=tiles+st*PART_BYTES')
    k=k.replace('y[(size_t)(off+n)*HD','y[((size_t)split*P+off+n)*HD')
    cu=cu[:b]+'\n'+k+cu[b:]
    cu=cu.replace('int e,nb,m0,m1,ne,off,task;','int e,nb,m0,m1,ne,off,task,split;')
    cu=cu.replace('int chunk,int hot,int cap){','int chunk,int hot,int cap,int sk,int cap2){')
    cu=cu.replace('tasks[67]=min(cap,u2);','tasks[67]=min(cap2,u2*sk);')
    cu=cu.replace('int64_t chunk,int64_t hot,int64_t cap){','int64_t chunk,int64_t hot,int64_t cap,int64_t sk,int64_t cap2){')
    cu=cu.replace('start,nt,chunk,hot,cap);','start,nt,chunk,hot,cap,sk,cap2);')
    cu+='''
size_t split_smem(int nt,int sk,int stages){int nw=sk<8?8/sk:1;return stages*(32768/sk+(sk==16?512:4096/sk))+nw*32*nt*4*sizeof(float)+128;}
int64_t split_capacity(int64_t nt,int64_t sk){int sm=split_smem(nt,sk,2),n;
 #define C(N,S) C10_CUDA_CHECK(cudaFuncSetAttribute(split_gemm_k<N,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,split_gemm_k<N,S>,(S<8?8/S:1)*32,sm));
 #define CS(N) if(sk==2){C(N,2);}else if(sk==4){C(N,4);}else if(sk==8){C(N,8);}else{C(N,16);}
 if(nt==1){CS(1);}else{CS(2);}
 #undef CS
 #undef C
 return n;
}
void split_gemm(torch::Tensor x,torch::Tensor bank,torch::Tensor gs,torch::Tensor inter,torch::Tensor y,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor plan,int64_t nt,int64_t chunk,int64_t sk,int64_t cap){
 auto stream=at::cuda::getCurrentCUDAStream();int sm=split_smem(nt,sk,2),P=x.size(0);
 #define R(N,S) split_gemm_k<N,S><<<cap,(S<8?8/S:1)*32,sm,stream>>>((const __nv_bfloat16*)x.data_ptr(),bank.data_ptr<uint8_t>(),gs.data_ptr<float>(),(__nv_bfloat16*)inter.data_ptr(),y.data_ptr<float>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),plan.data_ptr<int>(),1,chunk,2,P)
 #define RS(N) if(sk==2){R(N,2);}else if(sk==4){R(N,4);}else if(sk==8){R(N,8);}else{R(N,16);}
 if(nt==1){RS(1);}else{RS(2);}
 #undef RS
 #undef R
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
    tree=ast.parse(s)
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BAL_CUDA' for t in n.targets):n.value=ast.Constant(cu)
    s=ast.unparse(tree)+'\n'
    s=s.replace('muse_balance_','muse_balance_split_').replace("['balance_capacity', 'balance_route', 'balance_gemm']","['balance_capacity', 'balance_route', 'balance_gemm', 'split_capacity', 'split_gemm']")
    cpp=literal(s,'BAL_CPP').replace('int64_t,int64_t,int64_t,int64_t,int64_t);\nvoid balance_gemm','int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\nvoid balance_gemm')
    cpp+='''
int64_t split_capacity(int64_t,int64_t);
void split_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
'''
    tree=ast.parse(s)
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BAL_CPP' for t in n.targets):n.value=ast.Constant(cpp)
    s=ast.unparse(tree)+'\n'
    s=s.replace('return out\n\n@triton.jit','out[\'split_caps\']={(nt,sk):bank[\'grid\']*out[\'balanced_mod\'].split_capacity(nt,sk) for nt in (1,2) for sk in (2,4,8,16)}\n    return out\n\n@triton.jit',1)
    s=s.replace('def balanced_norm(Y, P, O, R, H: tl.constexpr=4096):','def balanced_norm(Y, P, O, R, P_MAX:tl.constexpr, SK:tl.constexpr, H: tl.constexpr=4096):')
    s=s.replace('v = tl.load(Y + p * H + k) * (1 + tl.load(P + k).to(tl.float32))',
'''v=tl.full((H,),0,tl.float32)
        for s in range(SK):v+=tl.load(Y+(s*P_MAX+p)*H+k)
        tl.store(Y+p*H+k,v)
        v *= 1+tl.load(P+k).to(tl.float32)''')
    s=s.replace('nt, chunk, hot = choice','nt, chunk, hot, sk = choice').replace('torch.empty((p, 4096), device=dev, dtype=torch.float32)','torch.empty((sk*p, 4096), device=dev, dtype=torch.float32)')
    s=s.replace("cap = bank['balanced_caps'][nt]","cap = bank['balanced_caps'][nt]\n    cap2=bank['split_caps'][nt,sk]")
    s=s.replace("nt, chunk, hot, cap)","nt, chunk, hot, cap,sk,cap2)")
    s=s.replace("mod.balance_gemm(inter, bank['w2p'], bank['gs2'], inter, y, counts, offs, order, pairs, plan, 1, nt, chunk // 2, 2, cap)","mod.split_gemm(inter, bank['w2p'], bank['gs2'], inter, y, counts, offs, order, pairs, plan, nt, chunk//2,sk,cap2)")
    s=s.replace("offs, inv, num_warps=4)","offs, inv, p,sk,num_warps=4)")
    s=s.replace("return f'b{c[0]}c{c[1]}h{c[2]}'","return f'k{c[3]}n{c[0]}c{c[1]}h{c[2]}'")
    a=s.index('BAL_CHOICES =');b=s.index('\n',a);s=s[:a]+"BAL_CHOICES = tuple((2,c,1,sk) for c in (32,64) for sk in (2,4,8,16)) + ((1,64,1,16),)"+s[b:]
    out=ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/split_w2.py';out.write_text(s);print(out)


if __name__=='__main__':build()
