"""Explicit sm120 admission: native DSM -> nonspinning global merge; native CAP1024 -> CAP512."""
import resgen

def install(ns,prefix=''):
    configs={'C2P':(4,True),'C4P':(4,True),'C8P':(8,True),'C16P':(16,True),'C4S':(4,False),'C8S':(8,False)}
    for n,(cs,pipe) in configs.items():
        ns[prefix+'KERNELS'][n]=resgen.source(cs,pipe,local=True)
        ns[prefix+'TOPOLOGY'][n]=cs
    return dict(reduction='genuine nonspinning global-last-CTA',C2='C4/CAP512 (sm120 shared limit)',other_caps='unchanged',WGMMA='exact mma.sync shim',native_performance_claim=False)
