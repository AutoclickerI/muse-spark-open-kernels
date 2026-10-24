#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cupti.h>
#include <cupti_activity.h>
#include <vector>
#include <tuple>
#include <string>
#include <mutex>
#include <cstdlib>
#include <stdexcept>

using ActivityRow=std::tuple<std::string,std::vector<int64_t>>;
namespace {
std::vector<ActivityRow> activity_rows;
std::mutex activity_mutex;
int activity_error=0;
size_t activity_dropped=0;
bool activity_on=false,activity_callbacks=false;
void ck(CUptiResult status){
 if(status!=CUPTI_SUCCESS){const char* text=nullptr;cuptiGetResultString(status,&text);throw std::runtime_error(text?text:"CUPTI activity error");}
}
void CUPTIAPI request_activity(uint8_t** buffer,size_t* size,size_t* max_records){
 *size=4<<20;*max_records=0;void* p=nullptr;
 if(posix_memalign(&p,64,*size))p=nullptr;
 *buffer=(uint8_t*)p;
}
void CUPTIAPI finish_activity(CUcontext context,uint32_t stream,uint8_t* buffer,size_t size,size_t valid){
 std::lock_guard<std::mutex> guard(activity_mutex);
 CUpti_Activity* record=nullptr;
 while(valid){
   CUptiResult status=cuptiActivityGetNextRecord(buffer,valid,&record);
   if(status==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
   if(status!=CUPTI_SUCCESS){activity_error=(int)status;break;}
   if(record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL||record->kind==CUPTI_ACTIVITY_KIND_KERNEL){
     auto* k=(CUpti_ActivityKernel10*)record;
     activity_rows.emplace_back(k->name?k->name:"",std::vector<int64_t>{
       (int64_t)k->start,(int64_t)k->end,(int64_t)k->queued,(int64_t)k->submitted,
       (int64_t)k->graphNodeId,(int64_t)k->gridId,(int64_t)k->graphId,
       k->gridX,k->gridY,k->gridZ,k->blockX,k->dynamicSharedMemory,k->registersPerThread});
   }
 }
 size_t lost=0;auto status=cuptiActivityGetNumDroppedRecords(context,stream,&lost);
 if(status!=CUPTI_SUCCESS)activity_error=(int)status;
 activity_dropped+=lost;free(buffer);
}
}

void profile_begin(){
 if(activity_on)throw std::runtime_error("activity recorder already enabled");
 if(!activity_callbacks){ck(cuptiActivityRegisterCallbacks(request_activity,finish_activity));activity_callbacks=true;}
 {std::lock_guard<std::mutex> guard(activity_mutex);activity_rows.clear();activity_error=0;activity_dropped=0;}
 ck(cuptiActivityEnableLatencyTimestamps(1));
 ck(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));activity_on=true;
}

std::vector<ActivityRow> profile_end(){
 if(!activity_on)throw std::runtime_error("activity recorder not enabled");
 auto flush_status=cuptiActivityFlushAll(0);
 auto disable_status=cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
 activity_on=false;
 auto latency_status=cuptiActivityEnableLatencyTimestamps(0);
 ck(flush_status);ck(disable_status);ck(latency_status);
 std::lock_guard<std::mutex> guard(activity_mutex);
 if(activity_error)ck((CUptiResult)activity_error);
 if(activity_dropped)throw std::runtime_error("activity recorder dropped records");
 return activity_rows;
}
