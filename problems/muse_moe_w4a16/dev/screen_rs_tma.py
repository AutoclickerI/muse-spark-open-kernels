"""Exercise the canonical TMA-weight pipeline without requiring Hopper WGMMA locally."""
from pathlib import Path
import sys
sys.argv=['screen_rs_tma.py','cand_rs_tma.py']
setup=(Path(__file__).parent/'screen_grouped.py').read_text().split("if '--tiles' in sys.argv:")[0]
exec(compile(setup,str(Path(__file__).parent/'screen_grouped.py'),'exec'))

xdata=R.generate_input(65,'ep8',828275)
for weight_stages in (1,2):
 st['weight_stages']=weight_stages
 for n in (8,16,32,64,128):
  needed=n*128*2+weight_stages*36864+32
  opt=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
  if needed>opt:
   print('HOPPER_ONLY N',n,'weight_stages',weight_stages,'shared bytes',needed,'local limit',opt,flush=True)
   continue
  out=c._grouped_call(xdata,st,n,2)
  torch.cuda.synchronize()
  print('N',n,'weight_stages',weight_stages,check(xdata,out),flush=True)
 del out
 torch.cuda.empty_cache()
# Force WS2 within the local memory limit; routing changes and multiple jobs test barrier reuse.
st['weight_stages']=2
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):out=c._grouped_call(xdata,st,64,2)
for distribution in ('hot','empty','normal','hot'):
 xdata['hidden_states'].normal_(generator=gen)
 if distribution=='hot':xdata['topk_ids'].copy_(torch.arange(8,device='cuda',dtype=torch.int32).expand(65,8))
 elif distribution=='empty':xdata['topk_ids'].fill_(60)
 else:xdata['topk_ids'].copy_(R.generate_input(65,'ep8',12442)['topk_ids'])
 graph.replay();torch.cuda.synchronize()
 print('WS2 graph routing',distribution,check(xdata,out),flush=True)
del graph,out,xdata
torch.cuda.empty_cache()
# Run the existing sampled large-M suite with the portable WS1 configuration.
st['weight_stages']=1
full=(Path(__file__).parent/'screen_grouped.py').read_text()
large=full.split("if '--large' in sys.argv:",1)[1]
exec(compile('if True:'+large,str(Path(__file__).parent/'screen_grouped.py'),'exec'))
