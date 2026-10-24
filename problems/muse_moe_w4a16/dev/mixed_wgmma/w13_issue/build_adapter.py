"""Self-contained pointer ABI, frozen raw baseline, and only W13 substitutions."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3]

def main():
 text=(R/'prof/lab/moe_w2fix_m256.py').read_text();t=ast.parse(text);base=next(n.value.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='FX_CUDA' for a in n.targets));(H/'base.cu').write_text(base)
 py=(P/'w2fix.py').read_text();t=ast.parse(py);t.body=[n for n in t.body if not (isinstance(n,ast.FunctionDef) and n.name=='fx_profile')]
 for n in t.body:
  if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='FX_CUDA' for a in n.targets):n.value=ast.parse("Path(__file__).with_name('base.cu').read_text()+Path(__file__).with_name('issue.cu').read_text()+Path(__file__).parents[1].joinpath('recode/recode.cu').read_text()",mode='eval').body
  if isinstance(n,ast.FunctionDef) and n.name=='fx_module':
   n.body=ast.parse('''target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
m=load_inline(name='w13_issue_'+hashlib.sha256(FX_CUDA.encode()).hexdigest()[:12],cpp_sources=FX_CPP+WI_CPP,cuda_sources=FX_CUDA,
functions=['fx_init','fx_info','fx_route','fx_run','fx_finish','wi_init','wi_info','wi_run','rc_init','rc_info','rc_run','wi_mixed','wi_mixed_info','wi_quad'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
m.fx_init();m.wi_init();m.rc_init();return m''').body
 py=ast.unparse(ast.fix_missing_locations(t));py=py.replace('pointers=vp2, M=m, cap=cap)','pointers=vp2, M=m, cap=cap, v13=v13)')
 assert 'v13=v13' in py
 extra='''
WI_CPP='int64_t wi_init();std::vector<int64_t> wi_info(int64_t);void wi_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);int64_t rc_init();std::vector<int64_t> rc_info(int64_t,int64_t,int64_t);void rc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void wi_mixed(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> wi_mixed_info(int64_t);void wi_quad(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'

def wi_prepare(weights):
    bank=fx_prepare(weights)
    for name in ('w13','w2'):
        q=bank[name];sf=bank[name+'_sf'];qr=torch.empty_like(q);sr=torch.empty_like(sf)
        rc_repack[(triton.cdiv(q.numel()//2,256),)](q,sf,qr,sr,q.numel()//2,RC_SIGNS,RC_ZEROS)
        bank[name+'_rec']=qr;bank[name+'_rec_sf']=sr
    bank['mix_stream']=torch.cuda.Stream(device=bank['w13'].device);bank['mix_start']=torch.cuda.Event();bank['mix_done']=torch.cuda.Event()
    return bank

def wi_plan(data,bank,policy=-1,grid=2112):
    plan=fx_plan(data,bank,1,grid,1)
    if policy==-1:return plan
    v=plan['v13'].copy();v2=plan['v2'].copy()
    if policy in (2,3,4,5,6,8,9):v[:2]=[bank['w13_rec'],bank['w13_rec_sf']]
    if policy in (3,4,5,6,8,9):v2[:2]=[bank['w2_rec'],bank['w2_rec_sf']]
    ptrs=[x.data_ptr() for x in v];p2=[x.data_ptr() for x in v2];m=plan['M'];cap=plan['cap'];start=plan['start']
    def first():
        _=v
        if policy in (7,8):
            bank['mod'].wi_quad(ptrs,0 if policy==7 else 1,grid,m,cap,start,torch.cuda.current_stream().cuda_stream)
        elif policy in (5,6,9):
            parent=torch.cuda.current_stream()
            if policy==5:
                bank['mod'].wi_mixed(ptrs,0,grid,m,cap,start,parent.cuda_stream);bank['mod'].wi_mixed(ptrs,1,grid,m,cap,start,parent.cuda_stream)
            else:
                bank['mix_start'].record(parent);bank['mix_stream'].wait_event(bank['mix_start'])
                with torch.cuda.stream(bank['mix_stream']):
                    if policy==9:bank['mod'].wi_quad(ptrs,2,grid,m,cap,start,bank['mix_stream'].cuda_stream)
                    else:bank['mod'].wi_mixed(ptrs,0,grid,m,cap,start,bank['mix_stream'].cuda_stream)
                    bank['mix_done'].record()
                if policy==9:bank['mod'].wi_quad(ptrs,3,grid,m,cap,start,parent.cuda_stream)
                else:bank['mod'].wi_mixed(ptrs,1,grid,m,cap,start,parent.cuda_stream)
                parent.wait_event(bank['mix_done'])
        elif policy==3:bank['mod'].rc_run(ptrs,1,1,grid,start,1,0,m,cap,torch.cuda.current_stream().cuda_stream)
        else:bank['mod'].wi_run(ptrs,2 if policy==4 else policy,grid,m,cap,start,torch.cuda.current_stream().cuda_stream)
    def second():
        _=v2
        bank['mod'].rc_run(p2,0,1,grid,start,1,0,m,cap,torch.cuda.current_stream().cuda_stream)
    steps=plan['steps'];plan['steps']=(steps[0],first,second if policy in (3,4,5,6,8,9) else steps[2],steps[3]);return plan

def wi_call(data,bank,policy=-1,grid=2112):
    for f in wi_plan(data,bank,policy,grid)['steps']:out=f()
    return out
'''
 pre='import triton\nfrom recoded import rc_repack,RC_SIGNS,RC_ZEROS\n'
 (H/'adapter.py').write_text(pre+py+'\n'+extra)
 print('adapter.py')

if __name__=='__main__':main()
