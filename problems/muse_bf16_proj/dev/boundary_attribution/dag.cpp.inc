// All capture nodes/edges, including non-kernel joins and programmatic edge types.
std::pair<std::vector<std::vector<int64_t>>,std::vector<std::vector<int64_t>>>
capture_dag(int64_t raw_stream) {
  cudaStreamCaptureStatus status;unsigned long long capture_id=0;cudaGraph_t graph=nullptr;
  auto e=cudaStreamGetCaptureInfo((cudaStream_t)raw_stream,&status,&capture_id,&graph,nullptr,nullptr);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  if(status!=cudaStreamCaptureStatusActive)throw std::runtime_error("DAG query requires active capture");
  size_t n=0;e=cudaGraphGetNodes(graph,nullptr,&n);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  std::vector<cudaGraphNode_t> nodes(n);e=cudaGraphGetNodes(graph,nodes.data(),&n);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  std::vector<std::vector<int64_t>> rows,edges;
  for(size_t i=0;i<n;i++) {
    cudaGraphNodeType type;e=cudaGraphNodeGetType(nodes[i],&type);
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    uint64_t id=0;
    if(type==cudaGraphNodeTypeKernel) {
      ck(cuptiGetGraphNodeId((CUgraphNode)nodes[i],&id));
    }
    rows.push_back({type==cudaGraphNodeTypeKernel?(int64_t)id:-(int64_t)(i+1),(int64_t)type});
  }
  size_t count=0;e=cudaGraphGetEdges(graph,nullptr,nullptr,nullptr,&count);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  std::vector<cudaGraphNode_t> from(count),to(count);std::vector<cudaGraphEdgeData> data(count);
  e=cudaGraphGetEdges(graph,from.data(),to.data(),data.data(),&count);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  for(size_t k=0;k<count;k++) {
    size_t a=0,b=0;while(a<n&&nodes[a]!=from[k])a++;while(b<n&&nodes[b]!=to[k])b++;
    if(a==n||b==n)throw std::runtime_error("DAG edge node missing");
    edges.push_back({(int64_t)a,(int64_t)b,data[k].type,data[k].from_port,data[k].to_port});
  }
  return {rows,edges};
}
