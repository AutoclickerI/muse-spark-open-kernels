// Create only our consumer node. The producer, its params, and all other
// graph work remain untouched. GDC.wait sees the full producer dependency.
inline void vr_check(cudaError_t e,const char* action) {
  if(e!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(e));
}
inline bool vr_capture_consumer(cudaStream_t stream,const cudaKernelNodeParams& params,int cluster,bool early) {
  if(!early)return false;
  cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;const cudaGraphNode_t* deps=nullptr;
  const cudaGraphEdgeData* data=nullptr;size_t count=0;unsigned long long id=0;
  vr_check(cudaStreamGetCaptureInfo(stream,&status,&id,&graph,&deps,&data,&count),"capture info");
  if(status==cudaStreamCaptureStatusNone)return false;
  if(status!=cudaStreamCaptureStatusActive || count!=1 || !graph)throw std::runtime_error("resident expects one captured QKV producer");
  cudaGraphNodeType type;vr_check(cudaGraphNodeGetType(deps[0],&type),"producer node type");
  if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("resident producer must be a kernel node");
  const cudaGraphNode_t producer=deps[0];cudaGraphNode_t consumer=nullptr;
  vr_check(cudaGraphAddKernelNode(&consumer,graph,nullptr,0,&params),"add resident consumer");
  cudaGraphEdgeData edge={};edge.type=cudaGraphDependencyTypeProgrammatic;edge.from_port=cudaGraphKernelNodePortLaunchCompletion;
  vr_check(cudaGraphAddDependencies(graph,&producer,&consumer,&edge,1),"add launch-completion dependency");
  if(cluster>=1) {
    cudaKernelNodeAttrValue attr={};attr.clusterDim.x=cluster;attr.clusterDim.y=attr.clusterDim.z=1;
    vr_check(cudaGraphKernelNodeSetAttribute(consumer,cudaKernelNodeAttributeClusterDimension,&attr),"resident cluster attribute");
  }
  cudaGraphEdgeData serial={};
  vr_check(cudaStreamUpdateCaptureDependencies(stream,&consumer,&serial,1,cudaStreamSetCaptureDependencies),"advance resident capture");
  return true;
}
