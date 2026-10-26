# Lab-only measurement. Every timing round contributes; no selective reruns or stopping.
S21_PROTOCOL={'name':'palindromic-cohort-v1','rounds':16,'seed_count':64,'aggregation':'arithmetic-mean-all-rounds','selective_retiming':False}
S21_TRACE_CONTEXT={};S21_ROUND_COUNTS={};S21_TOP_TRACES=[];S21_FLAGGED=0;S21_MEASURED=0
S21_NVML=None;S21_TELEMETRY_ERROR=None

def s21_telemetry():
    global S21_NVML,S21_TELEMETRY_ERROR
    import ctypes
    try:
        if S21_NVML is None:
            lib=ctypes.CDLL('libnvidia-ml.so.1');rc=lib.nvmlInit_v2()
            if rc:raise RuntimeError('nvmlInit_v2:'+str(rc))
            handle=ctypes.c_void_p();rc=lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0),ctypes.byref(handle))
            if rc:raise RuntimeError('nvmlDeviceGetHandleByIndex_v2:'+str(rc))
            S21_NVML=(lib,handle)
        lib,handle=S21_NVML
        def read(fn,*args):
            value=ctypes.c_uint();rc=getattr(lib,fn)(handle,*args,ctypes.byref(value))
            if rc:raise RuntimeError(fn+':'+str(rc))
            return value.value
        return dict(sm_MHz=read('nvmlDeviceGetClockInfo',ctypes.c_uint(1)),mem_MHz=read('nvmlDeviceGetClockInfo',ctypes.c_uint(2)),
                    temperature_C=read('nvmlDeviceGetTemperature',ctypes.c_uint(0)),power_mW=read('nvmlDeviceGetPowerUsage'),t_s=round(time.monotonic(),3))
    except Exception as exc:
        S21_TELEMETRY_ERROR=type(exc).__name__+':'+str(exc)
        return dict(error=S21_TELEMETRY_ERROR)

def s21_orders(names,rounds,seed_index):
    assert rounds>0 and rounds%2==0
    for block in range(rounds//2):
        shift=(seed_index+block)%len(names);order=names[shift:]+names[:shift]
        if (seed_index+block)%2:order=order[::-1]
        yield order
        yield order[::-1]

def ab_time(graphs,rounds):
    """Symmetric cohorts: ABBA with two arms, forward/reverse with more arms."""
    global S21_FLAGGED,S21_MEASURED
    assert rounds==S21_PROTOCOL['rounds']
    names=list(graphs);times={n:[] for n in names};enqueue={n:[] for n in names};telemetry=[s21_telemetry()]
    seed_index=S21_TRACE_CONTEXT['seed_index'];orders=list(s21_orders(names,rounds,seed_index))
    for r,order in enumerate(orders):
        evs=[]
        for n in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            lab_cold_l2()
            host_start=time.perf_counter_ns()
            start.record();graphs[n][0].replay();end.record()
            enqueue[n].append((time.perf_counter_ns()-host_start)/1000.)
            evs.append((n,start,end))
        torch.cuda.synchronize()
        for n,start,end in evs:times[n].append(start.elapsed_time(end)*1000.)
        if r%2:telemetry.append(s21_telemetry())
    label=S21_TRACE_CONTEXT['label'];counts=S21_ROUND_COUNTS.setdefault(label,{})
    for name,values in times.items():counts.setdefault(name,[]).append(len(values))
    # Preserve the worst four seed/shape traces, including all primary rounds
    # and before/after each symmetric block. This bounded diagnostic selection
    # never changes a timing value, a seed, an arm status or the G0 calculation.
    primary={n:v for n,v in times.items() if n in ('A','A2','INC')}
    spread=max(max(v)/statistics.median(v) for v in primary.values())
    aa=statistics.mean(primary['A2'])/statistics.mean(primary['A'])
    pair=statistics.mean(primary['INC'])/statistics.mean(primary['A'])
    severity=max(spread,math.exp(abs(math.log(aa))),math.exp(abs(math.log(pair))))
    flagged=spread>1.15 or abs(math.log(aa))>0.02 or abs(math.log(pair))>math.log(1.15)
    S21_FLAGGED+=int(flagged);S21_MEASURED+=1
    trace=dict(**S21_TRACE_CONTEXT,severity=severity,flagged=flagged,round_us=primary,
               host_enqueue_us={n:enqueue[n] for n in primary},orders=orders,telemetry=telemetry)
    S21_TOP_TRACES.append(trace);S21_TOP_TRACES.sort(key=lambda x:x['severity'],reverse=True);del S21_TOP_TRACES[4:]
    return times
