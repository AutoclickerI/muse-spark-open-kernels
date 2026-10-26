static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static int port_grid=0,port_smem=0;
void port_f(std::vector<int64_t>a,int64_t M,int64_t lo,int64_t stream){
 if(!port_grid){int d,mx;ck(cudaGetDevice(&d));ck(cudaDeviceGetAttribute(&mx,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));ck(cudaDeviceGetAttribute(&port_grid,cudaDevAttrMultiProcessorCount,d));
 ck(cudaFuncSetAttribute(moe_f,cudaFuncAttributeMaxDynamicSharedMemorySize,mx));
 #ifdef SMREQ
 if(SMREQ*1024<mx){mx=SMREQ*1024;ck(cudaFuncSetAttribute(moe_f,cudaFuncAttributePreferredSharedMemoryCarveout,(mx*100+228*1024-1)/(228*1024)));}
 #endif
 port_smem=mx;}
 P p{(uint8_t*)a[3],(uint8_t*)a[4],(float*)a[5],(float*)a[6],(__nv_bfloat16*)a[0],(int*)a[1],(float*)a[2],(float*)a[7],(float*)a[8],(__nv_bfloat16*)a[10],(int*)a[9],(int)M,(int)lo,port_smem,(unsigned long long*)a[11]};
 moe_f<<<port_grid,WPC*32,port_smem,(cudaStream_t)stream>>>(p);ck(cudaGetLastError());
}
