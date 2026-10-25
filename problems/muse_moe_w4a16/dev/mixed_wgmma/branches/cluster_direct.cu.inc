template<int NT,int J,int HALF>__device__ __forceinline__ void directx(
 __nv_bfloat16*bs,const __nv_bfloat16*root,int st,int rem,int src){
 int t=threadIdx.x,r=t/8+J*16,k=(t&7)*8;
 unsigned dst=mwbr::smaddr(bs+st*NT*128+HALF*NT*64+r*64+(k^((r&7)*8)));
 asm volatile("{ .reg .u64 p; .reg .u32 n; .reg .pred ok;"
              "setp.gt.s32 ok,%2,%4;selp.u32 n,16,0,ok;mad.wide.u32 p,%5,8192,%1;add.u64 p,p,%3;"
              "cp.async.cg.shared.global [%0],[p],16,n;}"
              ::"r"(dst),"l"(root),"r"(rem),"n"(HALF*64*2),"n"(J*16),"r"(src):"memory");
}
