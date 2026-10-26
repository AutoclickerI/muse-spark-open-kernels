"""Full exact packed batch, forced Hopper host glue, lossless bank/escape inverses."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[4]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
local=load('ng_local_standins',H/'local_standins.py');p=load('ng_packed_payload',H/'out/night_gap.py');p.PP_LOCAL_INSTALL=local.install
flush=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:flush.view(torch.int64).sum();raw=[];p.PP_LOCAL_ARCHIVE=raw.append;lines=p.lab(time.monotonic()+1500,None);out=raw[0];(H/'out/local_raw.json').write_text(json.dumps(out));assert out['complete'],out['fatal']
sys.path.insert(0,str(R/'tools'));from lab_packet import encode_packet,read_packets
wire=encode_packet('night_gap','0'*16,dict(lines=lines,complete=None,format='legacy-lines'));assert len(wire)+1<=13744,len(wire);assert len(read_packets((wire+'\n'+'suffix\n'*285+'X'*256)[-16000:]))==1
proof=dict(passed=True,payload_sha256=hashlib.sha256((H/'out/night_gap.py').read_bytes()).hexdigest(),layer_TASK=out['checks'],strict_TASK=out['strict_checks'],rows=len(out['rows']),round_counts=[r['rounds'] for r in out['rows']],host_launches=len(local.calls),codec_inverses=local.codec,wire_bytes=len(wire)+1,two_k_suffix_guard=True,limits='Hopper NF/DSM/TMA body uses explicit arithmetic stand-ins. Exact raw-pointer/cluster/PDL launch glue and packed weights/escape/table inverses exercised; native cross-compile separate, no H200 timing claim.')
(H/(sys.argv[1] if len(sys.argv)>1 else 'local_validation.json')).write_text(json.dumps(proof,indent=2)+'\n');print('PASS',out['checks'],out['strict_checks'],len(wire)+1,len(local.codec))
