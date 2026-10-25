"""Trace native preparation boundaries; arithmetic/timing kernels are unchanged."""
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'nf_port_lab.py').read_text()
# Replace the embedded adapter constant using AST literal evaluation only.
import ast
node=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='PORT_SOURCE' for t in n.targets))
p=ast.literal_eval(node.value)
p=p.replace("    for family,entries in grouped.items():", "    for family,entries in grouped.items():\n        print('NFTRACE family',family,flush=True)")
p=p.replace("            name=key[1];w=roles[name]", "            print('NFTRACE pack',key,flush=True)\n            name=key[1];w=roles[name]")
p=p.replace("    ns=state['ns'];dev=weights['o'].device", "    ns=state['ns'];dev=weights['o'].device\n    print('NFTRACE wide mw start',flush=True)")
p=p.replace("    # Raw BF16 SS WGMMA O shapes", "    print('NFTRACE wide mw done',flush=True)\n    # Raw BF16 SS WGMMA O shapes")
p=p.replace("        st=ns[build](weights,[cfg]);", "        print('NFTRACE wide O',t,flush=True)\n        st=ns[build](weights,[cfg]);")
p=p.replace("    ns['tt_v10_tune_RL_CUDA']", "    print('NFTRACE wide RL start',flush=True)\n    ns['tt_v10_tune_RL_CUDA']",1)
p=p.replace("    ns['tt_v14_qkv_QKVBIG_CUDA']", "    print('NFTRACE wide QB start',flush=True)\n    ns['tt_v14_qkv_QKVBIG_CUDA']",1)
p=p.replace("    lt=ns['tt_lt__lt_state'](weights)", "    print('NFTRACE wide LT start',flush=True)\n    lt=ns['tt_lt__lt_state'](weights)")
p=p.replace("    for t,name in NF_SMALL.keys() | NF_WIDE:", "    print('NFTRACE warm start',flush=True)\n    for t,name in sorted(NF_SMALL.keys() | NF_WIDE):\n        print('NFTRACE warm',t,name,flush=True)")
lines=s.splitlines(keepends=True);s=''.join(lines[:node.lineno-1])+'PORT_SOURCE = '+repr(p)+'\n'+''.join(lines[node.end_lineno:])
s=s.replace("        old['_seed_v9_gemv_choices']", "        print('NFTRACE control start',flush=True)\n        old['_seed_v9_gemv_choices']")
s=s.replace("        state['proj_plan']=", "        print('NFTRACE control coded done',flush=True)\n        state['proj_plan']=")
s=s.replace("            state['best7_stream_ops']", "            print('NFTRACE control SP start',flush=True)\n            state['best7_stream_ops']",1)
s=s.replace("            state['lt_mod']=", "            print('NFTRACE control LT start',flush=True)\n            state['lt_mod']=")
s=s.replace("            # Exact S13 short-LT", "            print('NFTRACE control LT done',flush=True)\n            # Exact S13 short-LT")
s=s.replace("            for t,nn in ((128,1),(256,0)):", "            print('NFTRACE control PG options start',flush=True)\n            for t,nn in ((128,1),(256,0)):\n                print('NFTRACE control PG',t,nn,flush=True)")
s=s.replace("        return state", "        print('NFTRACE control done',flush=True)\n        return state",1)
compile(s,'<trace lab>','exec')
(H/'nf_trace_lab.py').write_text(s)
