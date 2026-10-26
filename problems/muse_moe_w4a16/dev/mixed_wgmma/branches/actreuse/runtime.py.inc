ACT_NAMES=('S18','PACK','TMA1','MC2','MC4','W4N32','W4N64')
ACT_NS={}
def act_build():
 import linecache
 if ACT_NS:return
 ns=dict(globals());filename='<act-S18>';linecache.cache[filename]=(len(S18_SOURCE),None,S18_SOURCE.splitlines(True),filename);exec(compile(S18_SOURCE,filename,'exec'),ns)
 ns['p3b_module']();ns['p3b_module']().ch_info(4,0);ns['p3b_module']().ch_info(4,1);ns['mt_module']();ns['mt_pmz_module']();ns['NF_OPTS']['pmz']=ns['p3b_module']().pmz_init();ACT_NS['S18']=ns
 arm=dict(ns);filename='<act-cell>';linecache.cache[filename]=(len(ACT_CELL),None,ACT_CELL.splitlines(True),filename);exec(compile(ACT_CELL,filename,'exec'),arm);arm['activation_module']();ACT_NS['ARMS']=arm

def act_prepare(w):
 ns=ACT_NS['S18'];w=dict(w,post_norm_w=w['post_expert_norm'])
 base=ns['nf_perm'](ns['nf_wide'](ns['m128_hh_issue_prepare'](ns['nf_ilv'](w))));native=ns['m128_hh_issue_nt_prepare'](w)
 return dict(base=base,native=native,streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()))

def act_plan(d,s,arm):
 args=(d,s['base'],s['native'],s['streams'])
 if arm=='S18':return ACT_NS['S18']['p3b_plan'](*args)
 return ACT_NS['ARMS']['activation_plan'](*args,arm)
def act_call(d,s,arm):
 for f in act_plan(d,s,arm)['steps']:out=f()
 return out

def act_counts(d):
 import math
 m=d['hidden_states'].shape[0];cc=[int((d['topk_ids']==e).sum()) for e in range(32)];sms=132;grid=2*sms
 low=[n for n in cc if 0<n<=8];parts=[];low_bytes=0
 for nt in (512,256):
  U=len(low)*nt
  for j,n in enumerate(low):
   a=j*nt;b=a+nt;segments=sum(U*c//grid<U*(c+1)//grid and U*(c+1)//grid>a and U*c//grid<b for c in range(grid))
   low_bytes+=n*8192*segments
 heavy=sum(n for n in cc if n>16);medium=sum(n for n in cc if 8<n<=16)
 hx=heavy*96*8192;lx=low_bytes+medium*192*8192;gather=heavy*8192*2
 pad=sum(64*(n//64)+(32 if n%64 and n%64<=32 else 64 if n%64 else 0) for n in cc if n>16)*96*8192
 result={a:lx+hx for a in ACT_NAMES};result['PACK']+=gather
 for a,cs in (('TMA1',1),('MC2',2),('MC4',4)):result[a]=lx+pad/cs+gather
 for a in ('W4N32','W4N64'):result[a]=lx+hx/2
 weights={a:sum((math.ceil(n/64) if n>16 else int(n>0)) for n in cc)*28311552 for a in ACT_NAMES}
 weights['W4N32']=sum((math.ceil(n/32) if n>16 else int(n>0)) for n in cc)*28311552
 return dict(counts=cc,activation_MB={a:v/1e6 for a,v in result.items()},weight_MB={a:v/1e6 for a,v in weights.items()},heavy_valid_X_MB=hx/1e6,other_valid_X_MB=lx/1e6,gather_read_write_MB=gather/1e6,sector_utilization=1.0)
