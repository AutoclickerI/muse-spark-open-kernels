"""Source-backed PDL boundary inventory and prior production evidence."""
import ast,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/s21')]
from promotion import interval
from lab_packet import paired_evidence
import report
receipt=next(R.glob('runs/muse_bf16_proj/*5864563117fd44b6.json'))
raw=report.decode(paired_evidence(json.loads(receipt.read_text())['result']['log_tail'],'S21'))
prior={}
for load,row in raw['loads'].items():
 samples=row['samples'];aa=[s['A2'] for s in samples]
 prior[load]=dict(no_PUBLIC_over_S21=interval([s['no_PUBLIC'] for s in samples],aa),AA=interval(aa))
oldsrc=(H.parent/'public_ports/lee_kernels.cu.inc').read_text();oldhost=(H.parent/'public_ports/lee_host.cu.inc').read_text()
assert 'griddepcontrol' not in oldsrc+oldhost
pack=(H.parent/'public_ports/input.py.inc').read_text();assert 'launch_pdl' not in pack and 'gdc_' not in pack
rows=[]
for cell,dtype,s,width,kernel in [('d16S','FP32',10,1280,'Lee attnd'),('d32S','FP32',1,2304,'Lee attnd'),('v1S','FP32',1,2304,'OP _fused_attn'),('v1F','FP32',1,2304,'Lee attnp68'),('v16S','FP32',3,2304,'Lee attnp32836'),('v16F','FP32',3,2304,'Lee attnp324'),('v32S','BF16',1,2304,'Lee attnp32836')]:
 direct=cell in ('d32S','v1S');packing=dtype=='FP32' and not direct
 mechanism='ordinary launch blocks independent history producer/prologue and successor PDL overlap'
 if packing:mechanism+='; separate ordinary sum/cast launch'
 if cell=='d16S':mechanism+='; in particular the independent gate-split GEMM cannot start during attention'
 if cell=='v32S':mechanism+='; upstream Lt already completes ordinarily, so upstream overlap remains bounded'
 if cell in ('d32S','v1S'):mechanism+='; accepted direct-input cell already removed sum/cast'
 rows.append(dict(cell=cell,producer_ABI=[dtype,s,width],current_kernels=(['_pack_kernel'] if packing else [])+[kernel],current_wait='none: ordinary CUDA/Triton launch provides completion ordering',current_release='implicit kernel exit',extra_kernels=int(packing),
  successor='gate_split: entry release, waits only at final Y read' if cell=='d16S' else 'O at T128/T256: ordinary library completion boundary; native trace confirms exact backend' if cell.startswith(('v16','v32')) else 'owned NF O: entry release, dependency wait before activation reads',
  fixed_wait='pack: immediately before sum; Lee consumers after CTA initialization, producer gathers history without QKV wait; OP after scalar shape/seq/scale prologue',fixed_release='entry (PREEXIT in native Lee SASS)',
  fixed_kernels_D=(['_pack_kernel PDL'] if packing else [])+[kernel+' PDL'],fixed_kernels_F=[kernel+' direct-vector PDL'],source_mechanism=mechanism,timing_cause='unmeasured until D/V/F production graph contrasts land'))
out=dict(base='S21 eacc087b + 9b043487 = fbaf2793',prior_job='5864563117fd44b6',prior=prior,cells=rows,
 limitation='Prior no_PUBLIC isolates d16S/d32S/v32S only. v1 and v16 each replace both cells; those prior grouped ratios cannot identify the two cell effects. d32S and v1S changed in 9b043487, so prior points there are not the new baseline.',
 scope='Source proves missing PDL and extra nodes; it does not assign a measured number of lost microseconds. New lab profiles include attention, adjacent producer/successor nodes and full kernel_fn times.')
(H/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
for load,cell in [('decode 16','d16S'),('decode 32','d32S'),('verify 32','v32S')]:
 v=prior[load]['no_PUBLIC_over_S21'];print(cell,'prior port/S19',1/v['R'],[1/v['CI95'][1],1/v['CI95'][0]])
