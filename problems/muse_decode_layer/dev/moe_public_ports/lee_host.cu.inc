static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static int port_grid=0;
static size_t port_smem(int M){return (size_t)RING+2*NSL*8+sizeof(Sm)+(size_t)M*8*2+1024;}
void port_x(std::vector<int64_t> a,int64_t M,int64_t lo,int64_t mode,int64_t stream){
 auto st=(cudaStream_t)stream;
 if(!port_grid){int d,mx;ck(cudaGetDevice(&d));ck(cudaDeviceGetAttribute(&mx,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));ck(cudaDeviceGetAttribute(&port_grid,cudaDevAttrMultiProcessorCount,d));
 ck(cudaFuncSetAttribute(moe_x<-1>,cudaFuncAttributeMaxDynamicSharedMemorySize,mx));
 ck(cudaFuncSetAttribute(moe_x<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,mx));
 ck(cudaFuncSetAttribute(moe_x<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,mx));}
 if(M*8>4096 || M*8*3+4096>RING)throw std::runtime_error("M too large");
 P p{(uint8_t*)a[3],(uint8_t*)a[4],(float*)a[5],(float*)a[6],(uint8_t*)a[8],(uint8_t*)a[9],(int*)a[1],(float*)a[2],(float*)a[7],(__nv_bfloat16*)a[10],(float*)a[11],(__nv_bfloat16*)a[13],(int*)a[12],(int)M,(int)lo,(unsigned long long*)a[14]};
 if(mode==0)xgather<<<M,256,0,st>>>((__nv_bfloat16*)a[0],(int*)a[1],M,lo,(uint8_t*)a[8]);
 else if(mode==1)moe_x<-1><<<port_grid,NTH,port_smem(M),st>>>(p);
 else if(mode==2)moe_x<0><<<port_grid,NTH,port_smem(M),st>>>(p);
 else if(mode==3){ck(cudaMemsetAsync((void*)a[11],0,M*8*sizeof(float),st));moe_x<1><<<port_grid,NTH,port_smem(M),st>>>(p);}
 else if(mode==4)finish_x<<<port_grid,NTH,0,st>>>(p);
 else throw std::runtime_error("bad mode");
 ck(cudaGetLastError());
}
