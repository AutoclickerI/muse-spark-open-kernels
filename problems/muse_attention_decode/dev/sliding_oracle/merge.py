"""Same RAW2 slices/math, nonspinning last-WG reduction through GPU memory.

Two independent WGs per CTA publish four logical slices. The last publisher
alone reduces; no CTA spins waiting for an unscheduled block.
"""
import structural
from pathlib import Path
H=Path(__file__).resolve().parent

def ordering(s):
    s='#include <cuda/atomic>\n'+s
    old='atomicAdd(p.count+(b*p.HK+kh)*32,1)';assert s.count(old)==1
    s=s.replace(old,'cuda::atomic_ref<int,cuda::thread_scope_device>(p.count[(b*p.HK+kh)*32]).fetch_add(1,cuda::memory_order_acq_rel)')
    old='atomicExch(p.count+(b*p.HK+kh)*32,0)';assert s.count(old)==1
    s=s.replace(old,'cuda::atomic_ref<int,cuda::thread_scope_device>(p.count[(b*p.HK+kh)*32]).store(0,cuda::memory_order_release)')
    old='ml[c]=*(float2*)(pp+64);ov[c]=*(float2*)(pp+2*lane);';assert s.count(old)==1
    s=s.replace(old,'ml[c]=__ldcg((const float2*)(pp+64));ov[c]=__ldcg((const float2*)(pp+2*lane));')
    return s

def source(local=False):
    # This former local DSM substitute is now an intentional native arm.
    # Streaming geometry is unchanged by local=True: CCTA2/WG2/4 slices.
    return ordering(structural.oracle.window.source(2,stream=True,local=True))

def context_clock():
    return ordering(structural.context_clock('v',False,True))
