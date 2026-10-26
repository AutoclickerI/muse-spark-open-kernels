_SC_MOD=[]
def sc_module():
    if not _SC_MOD:
        arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
        _SC_MOD.append(load_inline(name='p7_cont_'+hashlib.sha256(SC_CUDA.encode()).hexdigest()[:12],
            cpp_sources='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nvoid sc_run(std::vector<int64_t>,int64_t,int64_t,int64_t);',
            cuda_sources=SC_CUDA,functions=['sc_run'],no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False))
    return _SC_MOD[0]

def sc_bank(state,plan,t,name):
    # Mirror actual projection dispatch. Original BF16 matrices are not the
    # consumer's bank when an NF, coded or explicitly packed layout is selected.
    key=(t,name)
    if name=='pool_router' and t in globals().get('P10_POOL_TOKENS',()):
        p=state['p10_pool_raw'];return [('S17 pool NF wp',p['wp'],1),('S17 pool NF metadata',p['esc'],1)]
    if key in globals().get('LIBRARY_NATIVE_ROWS',{}):return [('VDSK raw',state[name],state[name].shape[0])]
    if key in NF_PORT_SELECTED:
        nf=state['nf_port'];p=nf['plans'].get(key)
        if p is None:
            assert not nf['native'],'missing native NF plan'
            return [('local NF arithmetic stand-in',nf['roles'][name],nf['roles'][name].shape[0])]
        kind=p.get('kind','small')
        if kind in ('small','mw'):
            # Warp-stream packets are K-major across all warp groups. A flat
            # prefix covers the first pipeline planes, unlike raw row-major W.
            return [('NF '+kind+' wp',p['wp'],1),('NF '+kind+' metadata',p['esc'],1)]
        if kind=='ol':
            st=p['state'];fmt=st['configs'][0][4];w=st['tiled32'] if fmt==2 else st['packed'] if fmt==1 else st['weight']
            return [('NF ol packed'+str(fmt),w,w.shape[0])]
        if kind in ('rl','qb'):return [('NF '+kind+' packed',p['w'],p['w'].shape[0]//64)]
        assert kind=='lt',kind
        layout=nf['ns']['tt_lt__LT_H200_FIXED'][key][0]
        w=nf['lt']['packed'][name] if layout>=2 else nf['lt']['weights'][name]
        return [('NF LT layout'+str(layout),w,w.shape[0])]
    if key in CD_ROWS:
        c=state['coded_dense'][name];splits=CD_ROWS[key][3];kb=c['K']//64;rows=c['N']//16
        return [('CD split'+str(s),c['coded'],rows,s*kb//splits*1536,((s+1)*kb//splits-s*kb//splits)*1536,kb*1536) for s in range(splits)]+[
            ('CD scale',c['scales'],1),('CD eoff',c['eoff'],1),('CD escapes',c['esc'],1)]
    choice=plan['projection'].get(name,'auto')
    if choice=='auto':choice=state['proj_plan'].get(key,'custom')
    if isinstance(choice,tuple) and choice[0]=='short_lt':
        w,_=state['measured_lt']['layouts'][(t,name,choice[1])]
        return [('measured LT layout'+str(choice[1]),w,w.shape[0])]
    return [('raw BF16',state[name],state[name].shape[0])]

def sc_regions(state,plan,t,name,mb,flat=False):
    banks=sc_bank(state,plan,t,name);big=sum(1 for z in banks if z[1].numel()*z[1].element_size()>1<<20)
    big=max(big,1);descs=[];audit=[]
    for row in banks:
        label,w,groups=row[:3];size=w.numel()*w.element_size()
        assert w.is_contiguous(),label
        offset=0;stride=size//groups;available=stride
        if len(row)>3:offset,available,stride=row[3:]
        budget=(mb<<20)//big if size>1<<20 else min(size,256<<10)
        if flat and len(row)==3:groups=1;stride=size;available=size
        head=min(available,(budget//groups)//128*128)
        if head==0:head=min(available,128)
        head=head//16*16
        if not head:continue
        assert groups*head<=size and offset+(groups-1)*stride+head<=size,(label,size,offset,groups,stride,head)
        assert offset%16==stride%16==0,label
        descs.extend([w.data_ptr()+offset,groups,stride,head])
        audit.append(dict(bank=label,shape=list(w.shape),dtype=str(w.dtype),storage_bytes=size,
                          groups=groups,stride_bytes=stride,head_bytes=head,prefetch_bytes=groups*head,
                          consumer_pointer=w.data_ptr(),offset=offset))
    assert len(descs)<=64,(name,len(descs))
    return descs,audit

_SC_CURRENT_FUTURE=()
SC_ACTIVE=False
_sc_base_run=_best7_run
def _best7_run(data,state,plan,prefetch=None):
    global _SC_CURRENT_FUTURE
    if not SC_ACTIVE:return _sc_base_run(data,state,plan,prefetch)
    assert prefetch is None,'pre-existing prefetch must be independently composed'
    t=data['config']['tokens'];role='unpooler_slice' if t<=8 else 'unpooler'
    late=SC_ARM in ('P','X') and t<=32
    _SC_CURRENT_FUTURE=tuple(sc_regions(state,plan,t,role,4)[0]) if late else ()
    assert len(_SC_CURRENT_FUTURE)<=8
    if SC_ARM in ('F','B','L','X'):
        mode={'F':0,'B':1,'L':2,'X':1}[SC_ARM];mb=16 if SC_ARM=='L' else 4
        class Op:
            def begin(self,phase):
                name={'front':'o','pool':'pool_router','tail':role}[phase]
                if phase=='tail' and late:return
                args,_=sc_regions(state,plan,t,name,mb,flat=mode==2)
                sc_module().sc_run(args,state['sc_sink'].data_ptr(),mode,torch.cuda.current_stream().cuda_stream)
            def join(self,phase):pass
        prefetch=Op()
    try:return _sc_base_run(data,state,plan,prefetch)
    finally:_SC_CURRENT_FUTURE=()

_sc_base_prepare=prepare
def prepare(weights):
    global SC_ACTIVE
    state=_sc_base_prepare(weights)
    state['sc_sink']=torch.empty(8*128,device=weights['o'].device,dtype=torch.int32)
    sc_module();SC_ACTIVE=True;return state
