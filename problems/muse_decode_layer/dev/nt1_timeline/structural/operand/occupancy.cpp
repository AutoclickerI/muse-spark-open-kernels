#include <cuda_occupancy.h>
#include <cstdio>
#include <cstdlib>
// H200 resource model, checked again by runtime occupancy in the receipt.
int main(int argc, char** argv) {
  if (argc != 4) return 2;
  cudaOccDeviceProp p;
  p.computeMajor=9; p.computeMinor=0; p.maxThreadsPerBlock=1024;
  p.maxThreadsPerMultiprocessor=2048; p.regsPerBlock=65536;
  p.regsPerMultiprocessor=65536; p.warpSize=32; p.sharedMemPerBlock=49152;
  p.sharedMemPerMultiprocessor=233472; p.numSms=132;
  p.sharedMemPerBlockOptin=232448; p.reservedSharedMemPerBlock=1024;
  cudaOccFuncAttributes a;
  a.maxThreadsPerBlock=1024; a.numRegs=std::atoi(argv[1]);
  a.sharedSizeBytes=std::atoi(argv[3]); a.shmemLimitConfig=FUNC_SHMEM_LIMIT_OPTIN;
  a.maxDynamicSharedSizeBytes=232448-a.sharedSizeBytes;
  cudaOccDeviceState s;
  cudaOccResult r{};
  auto e=cudaOccMaxActiveBlocksPerMultiprocessor(&r,&p,&a,&s,256,std::atoi(argv[2]));
  if (e) return e;
  std::printf("%d\n",r.activeBlocksPerMultiprocessor);
}
