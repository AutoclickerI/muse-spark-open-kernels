import math
import torch
from cuda.bindings import driver as mc_driver


class MCDriverError(RuntimeError):
    def __init__(self,operation,status):
        self.operation=operation;self.status=int(status)
        super().__init__(f'{operation}: CUDA driver status={int(status)}')


def mc_call(operation,*args):
    result=getattr(mc_driver,operation)(*args)
    if int(result[0]):raise MCDriverError(operation,result[0])
    return result[1] if len(result)==2 else result[1:]


def mc_property(device,compression):
    prop=mc_driver.CUmemAllocationProp()
    prop.type=mc_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type=mc_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id=int(device);prop.allocFlags.compressionType=int(compression)
    return prop


class MCArray:
    def __init__(self,owner,offset,size):
        self.owner=owner
        self.__cuda_array_interface__=dict(shape=(size,),strides=None,typestr='|u1',
            data=(int(owner.pointer)+offset,False),version=3,stream=1)


class MappedAllocation:
    """Prepare-only driver allocation. Keep its owner with the weight state.

    close() requires completed GPU work and no live graph/tensor users. It is
    explicit so GC can never unmap a captured kernel's weights asynchronously.
    """
    def __init__(self,size,compression,device=None):
        self.device=torch.cuda.current_device() if device is None else int(device)
        self.logical_size=int(size);self.requested=int(compression)
        self.handle=None;self.pointer=None;self.mapped=False;self.closed=False
        prop=mc_property(self.device,compression)
        self.granularity=mc_call('cuMemGetAllocationGranularity',prop,
            mc_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
        self.size=math.ceil(size/self.granularity)*self.granularity
        try:
            self.handle=mc_call('cuMemCreate',self.size,prop,0)
            granted=mc_call('cuMemGetAllocationPropertiesFromHandle',self.handle)
            self.granted=int(granted.allocFlags.compressionType)
            self.pointer=mc_call('cuMemAddressReserve',self.size,self.granularity,0,0)
            mc_call('cuMemMap',self.pointer,self.size,0,self.handle,0);self.mapped=True
            access=mc_driver.CUmemAccessDesc();access.location=prop.location
            access.flags=mc_driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            mc_call('cuMemSetAccess',self.pointer,self.size,[access],1)
        except Exception:
            self.close();raise

    def tensor(self,shape,dtype=torch.uint8,offset=0):
        size=math.prod(shape)*torch.empty((),dtype=dtype).element_size()
        if self.closed or offset<0 or offset+size>self.logical_size:raise ValueError('mapped view exceeds allocation')
        proxy=MCArray(self,offset,size)
        byte_tensor=torch.as_tensor(proxy,device=torch.device('cuda',self.device))
        if byte_tensor.data_ptr()!=int(self.pointer)+offset:raise RuntimeError('mapped tensor was copied')
        return byte_tensor.view(dtype).reshape(shape)

    def close(self):
        if self.closed:return
        if self.mapped:mc_call('cuMemUnmap',self.pointer,self.size);self.mapped=False
        if self.pointer is not None:mc_call('cuMemAddressFree',self.pointer,self.size);self.pointer=None
        if self.handle is not None:mc_call('cuMemRelease',self.handle);self.handle=None
        self.closed=True


def mc_probe():
    device=torch.cuda.current_device()
    status,value=mc_driver.cuDeviceGetAttribute(
        mc_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,device)
    report=dict(attribute_status=int(status),attribute=int(value),requested=1)
    prop=mc_property(device,1)
    status,gran=mc_driver.cuMemGetAllocationGranularity(prop,
        mc_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
    report['granularity_status']=int(status)
    # Still attempt cuMemCreate if this property query fails: support is an
    # observation, not a reason to omit the requested API call.
    if int(status):
        gran=mc_call('cuMemGetAllocationGranularity',mc_property(device,0),
            mc_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
    size=math.ceil((2<<20)/gran)*gran
    result=mc_driver.cuMemCreate(size,prop,0);report['create_status']=int(result[0])
    report.update(probe_bytes=size,granularity=int(gran),properties_status=None,granted=None)
    if int(result[0])==0:
        handle=result[1]
        result=mc_driver.cuMemGetAllocationPropertiesFromHandle(handle)
        report['properties_status']=int(result[0])
        if int(result[0])==0:report['granted']=int(result[1].allocFlags.compressionType)
        mc_call('cuMemRelease',handle)
    return report
