"""Contiguous K-major TMA weight tiles and arbitrary disjoint K splits."""
import hashlib
import ss_source
CONFIGS=[(t,bw,1,q,pre) for t in (128,256) for bw,q,pre in ((64,2,0),(64,4,0),(64,5,0),(128,2,0),(128,4,0),(128,2,1))]

def source(configs=CONFIGS):
    s=ss_source.source(configs)
    old='tma(dest+XS,&tw,kk,nt*BW,full+slot*8,pw);';new='tma(dest+XS,&tw,0,((kb0+i*KG+g)*(n/BW)+nt)*BW,full+slot*8,pw);';assert s.count(old)==1;s=s.replace(old,new)
    old='rc=map(&tw,(void*)w,n,k,BW);';new='rc=map(&tw,(void*)w,n*(k/64),64,BW);';assert s.count(old)==1;s=s.replace(old,new)
    old='if(m!=BT || n%BW || k%(64*s*KG))return -1;';new='if(m!=BT || n%BW || k%64 || s<1 || (KG>1 && k%(64*s*KG)))return -1;';assert s.count(old)==1;s=s.replace(old,new)
    return s.replace('p10_ss_','p10_ps_')
