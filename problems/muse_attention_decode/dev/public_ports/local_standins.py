"""Local stand-ins preserve selected glue; native code is separately compiled for sm90."""
import torch
class GluonMath:
 def __init__(self,m,length):self.m=m;self.length=length
 def __getitem__(self,grid):
  def run(*a,**kw):
   qkv,loc,ks,vs,kc,vc,pt,seq,po,ml,cnt,out,rs,pts,chunk,scale,eps=a
   B=grid[0];L=self.length;ns=kw['NSPLIT'];rows=16 if L==1 else 64;hpb=2 if L==1 else 1;units=B if L==1 else B*2
   self.m._fused_attn[(B,2//hpb,ns)](qkv,loc,qkv,ks,vs,kc,vc,pt,seq,po,ml,po,ml,cnt,out,rs,pts,chunk,scale,eps,HQ=16,HKV=2,D=64,L=L,HPB=hpb,ROWS=rows,LP=16,PAGE=1,BLOCK_N=128,NSPLIT=ns,NSP2=kw['NSP2'],SB=1,WINDOW=0,ROPE=False,GS=16,NG=1,NG2=1,PREFETCH=False,NRB=1,QK8=False,GPRE=False,num_warps=4,num_stages=2)
  return run
def operator(m,forced):
 m._HOPPER_FORCE=forced
 for cfg in m.FUSED_CFG.values():
  cfg['stages']=2
  if 'small' in cfg:cfg['small']['stages']=2
  if 'big' in cfg:cfg['big']['stages']=2
 if forced:m._g_full_decode=GluonMath(m,1);m._g_full_verify=GluonMath(m,8)
def lee(m):m._LOCAL=True
