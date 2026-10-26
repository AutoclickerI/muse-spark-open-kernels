"""Compile actual sm90a source; retain ordered dependency loads and resources."""
import hashlib,json,pathlib,re,subprocess
H=pathlib.Path(__file__).resolve().parent;d=H/'native';d.mkdir(exist_ok=True);s=(H/'native_generated.py.inc').read_text();p=d/'residual.cu';p.write_text(s)
subprocess.run(['nvcc','-std=c++20','-O3','--expt-relaxed-constexpr','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(p.with_suffix('.o'))],check=True)
sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);usage=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass);p.with_suffix('.resources').write_text(usage);rows=[]
for block in sass.split('Function : ')[1:]:
 symbol=block.splitlines()[0];ins=[l.strip() for l in block.splitlines() if re.search(r'/\*[0-9a-f]+\*/',l)];waits=[i for i,l in enumerate(ins) if 'ACQBULK' in l]
 if not waits:continue
 assert not any(re.search(r'\b(LDL|STL)\b',l) for l in ins),symbol
 assert not any('LDG.' in l and 'STRONG.GPU' in l for l in ins[:waits[0]]),('dependent CG load before wait',symbol)
 rows.append(dict(symbol=symbol,waits=waits,ordered_loads_after_wait=sum('LDG.' in l and 'STRONG.GPU' in l for l in ins[waits[0]:]),stack_or_local_sass=False))
assert len(rows)==1 and len(rows[0]["waits"])==2,len(rows)
w=rows[0]['waits'][-1]
loads=[line for line in ins[w-20:w] if 'LDG.' in line]
post=[line for line in ins[w+1:w+17] if 'LDG.' in line]
assert len(loads)==2 and all('128.STRONG.GPU' in line for line in loads)
assert len(post)==2
# A CPU-only resource probe ranks load ordering; it is never a runtime arm.
anchor='      float4 tw=rt_load4(prew+j),ta=rt_load4(alpha+j);\n      pdl_wait();'
assert s.count(anchor)==1
after=s.replace(anchor,'      pdl_wait();\n      float4 tw=rt_load4(prew+j),ta=rt_load4(alpha+j);')
probe=d/'after_wait_probe.cu';probe.write_text(after)
subprocess.run(['nvcc','-std=c++20','-O3','--expt-relaxed-constexpr','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-c',str(probe),'-o',str(probe.with_suffix('.o'))],check=True)
other=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(probe.with_suffix('.o'))],text=True)
choice=dict(selected_pre_wait_regs=int(re.search(r'REG:(\d+)',usage)[1]),rejected_after_wait_regs=int(re.search(r'REG:(\d+)',other)[1]),rejected_source_sha256=hashlib.sha256(after.encode()).hexdigest(),rejected_resource_usage=other,local_SASS_absent=True)
order=dict(immutable_weights_before_wait=loads,wait=ins[w],dependent_stream_and_stats_after_wait=post,proof='Explicit PTX and tail-branch SASS; prefix path has its own ACQBULK')
(H/'native_audit.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(s.encode()).hexdigest(),kernels=rows,resources=usage,tail_SASS_order=order,resource_choice=choice),indent=2)+'\n')
print('PASS sm90a dedicated b2 tail/prefix specialization;2 immutable loads before tail wait,2 dependent loads after;registers',choice['selected_pre_wait_regs'],choice['rejected_after_wait_regs'])
