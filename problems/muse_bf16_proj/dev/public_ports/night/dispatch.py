"""One-shot dependency queue: parent sender completion -> exact receipt -> h200_lab.

No polling, delay, direct judge API, signal to another process, or background git writes.
The worker stops after one next send; it never keeps posting recommendations.
"""
import ctypes,hashlib,importlib.util,json,os,select,struct,subprocess,sys,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=H.parents[4];Q=H/'queue.json'
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def save(status,**fields):
 old=json.loads(Q.read_text()) if Q.exists() else {};old.update(status=status,**fields);tmp=Q.with_name(Q.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(old,indent=2)+'\n');tmp.replace(Q)
def pidfd_open(pid):
 # Host Python/glibc expose no pidfd wrapper; use the x86_64 Linux ABI directly.
 import platform
 assert platform.machine()=='x86_64'
 fn=ctypes.CDLL(None,use_errno=True).syscall;fn.restype=ctypes.c_long;fd=fn(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
 if fd<0:raise OSError(ctypes.get_errno(),'SYS_pidfd_open')
 return fd
def await_receipt(receipt,job):
 # A concurrent collector may claim the result just before the original sender
 # exits. Register one short-lived filesystem event watch before checking it.
 lib=ctypes.CDLL(None,use_errno=True);lib.inotify_init1.argtypes=[ctypes.c_int];lib.inotify_init1.restype=ctypes.c_int;lib.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];lib.inotify_add_watch.restype=ctypes.c_int
 fd=lib.inotify_init1(os.O_CLOEXEC)
 if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
 try:
  wd=lib.inotify_add_watch(fd,os.fsencode(receipt.parent),8|128)
  if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
  if receipt.exists():json.loads(receipt.read_text());return receipt
  subprocess.run([sys.executable,str(R/'tools/okc_run.py'),'collect',job],cwd=R,check=True)
  if receipt.exists():json.loads(receipt.read_text());return receipt
  while True:
   select.select([fd],[],[]);data=os.read(fd,65536);pos=0
   while pos<len(data):
    event,mask,cookie,length=struct.unpack_from('iIII',data,pos);name=data[pos+16:pos+16+length].split(b'\0',1)[0];pos+=16+length
    if name==os.fsencode(receipt.name) and mask&(8|128):json.loads(receipt.read_text());return receipt
 finally:os.close(fd)
def parent_receipt():
 wanted=json.loads((P/'queue.json').read_text())['sender']['sha256']
 for path in (R/'runs/muse_bf16_proj').glob('*.py'):
  if hashlib.sha256(path.read_bytes()).hexdigest()==wanted:
   receipt=path.with_suffix('.json')
   return await_receipt(receipt,path.stem.rsplit('_',1)[1])
 return None
def work():
 save('validating fixed one-shot queue',worker_pid=os.getpid());admission=json.loads((H/'admission.json').read_text());assert admission['passed'];parent=json.loads((P/'queue.json').read_text())['sender'];snapshot=R/'runs/sent'/parent['sha256'][:16]/Path(parent['file']).name;assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==parent['sha256']
 receipt=parent_receipt()
 if receipt is None:
  fd=pidfd_open(parent['pid'])
  try:
   cmd=Path(f"/proc/{parent['pid']}/cmdline").read_bytes();assert b'okc_run.py' in cmd and b'lab_public_whole.py' in cmd,'parent PID identity changed'
   save('waiting on parent sender completion event',dependency=parent,worker_pid=os.getpid(),one_shot=True,send_via='tools/h200_lab.py normal sender; --lane P10; PLAN rank',fixed_code_sha256=admission['code_template_sha256']);print('WAIT parent sender',parent['pid'],'pidfd completion',flush=True);select.select([fd],[],[])
  finally:os.close(fd)
  receipt=parent_receipt()
 assert receipt is not None,'parent sender exited without an exact complete receipt'
 save('reading exact parent native receipt',parent_receipt=str(receipt.relative_to(R)))
 subprocess.run([sys.executable,str(P/'whole_read.py'),str(receipt)],cwd=R,check=True)
 result=json.loads((P/'whole_results.json').read_text());portfolio=load('ng_parent_portfolio',H/'portfolio.py');selection=portfolio.select_parent(result)
 allowed=json.loads((P/'whole_manifest.json').read_text());assert all(p in allowed['points']['PORTS'] for p in selection['points']);assert all(a in allowed['pool'] for a in selection['pool'].values())
 (H/'parent_selection.json').write_text(json.dumps(selection,indent=2)+'\n');(P/'in_graph_points.json').write_text(json.dumps(selection['points'],indent=2)+'\n')
 source=subprocess.check_output(['git','show',admission['S21_git_blob']],cwd=R).decode();assert hashlib.sha256(source.encode()).hexdigest()==allowed['base_sha256']
 parent_candidate=portfolio.apply(source,selection);(H/'out/parent_nominee.py').write_text(parent_candidate)
 handoff=json.loads((P/'P4_HANDOFF.json').read_text());handoff.update(patch='problems/muse_bf16_proj/dev/public_ports/night/portfolio.py',command='python3 problems/muse_bf16_proj/dev/public_ports/night/portfolio.py BASE.py OUTPUT.py problems/muse_bf16_proj/dev/public_ports/night/parent_selection.json',points='problems/muse_bf16_proj/dev/public_ports/in_graph_points.json',priority_points='problems/muse_bf16_proj/dev/public_ports/in_graph_points.json',status='Native in-graph parent results available; stage-only losers dropped. P4 new composition still needs held-out G0.',in_graph_receipt=result['job'],in_graph_selection=selection,in_graph_patch=dict(script='problems/muse_bf16_proj/dev/public_ports/night/portfolio.py',function='apply(BASE_SOURCE, selection)',selection='problems/muse_bf16_proj/dev/public_ports/night/parent_selection.json',generated_source=str((H/'out/parent_nominee.py').relative_to(R)),sha256=hashlib.sha256(parent_candidate.encode()).hexdigest()),next_lab='night: seven strongest carriers with leave-outs; no new native performance claim before its receipt')
 (P/'P4_HANDOFF.json').write_text(json.dumps(handoff,indent=2)+'\n')
 builder=load('ng_fixed_builder',H/'build.py');manifest=builder.build(selection);guard=load('ng_admission_guard',H/'admit.py');assert manifest['source_sha256']==admission['source_sha256'];assert manifest['body_sha256']==admission['body_sha256'];assert guard.code_hash((H/'out/night_gap.py').read_text())==admission['code_template_sha256'],'unreviewed code change after local admission'
 save('sending next lab through h200_lab',parent_job=result['job'],selection=selection,payload_sha256=manifest['payload_sha256'])
 subprocess.run([sys.executable,str(P.parent/'native_boundary/send.py'),'send',str(H/'out/night_gap.py'),'--host','muse_bf16_proj','--lab-only','--import-budget','840','--lane','P10','--decision','small-M carrier portfolio and seven leave-outs after 3347492','--note','After exact3347492 native receipt: selected S21 port/pool portfolio; Q16/Q32/O16/P16/P32/U16/U32 coded DSM and tiled-TMA early weights;8disjoint seeds,all cyclic+reverse2N rounds,TASK every arm,selected prepare costs.'],cwd=R,check=True)
 wrapper=R/'problems/muse_bf16_proj/lab/lab_night_gap.py';sha=hashlib.sha256(wrapper.read_bytes()).hexdigest();entries=[]
 for path in (R/'runs/waiting').glob('*.json'):
  data=json.loads(path.read_text())
  if data.get('sha256')==sha:entries.append(data)
 save('next lab queued; one-shot worker complete',wrapper_sha256=sha,next_senders=entries);print('DONE next h200 sender',entries,flush=True)
def start():
 assert not Q.exists(),'A night queue already exists; inspect it before creating another worker'
 (H/'out').mkdir(exist_ok=True);save('dependency dispatcher starting',parent_sender=json.loads((P/'queue.json').read_text())['sender']['pid'],parent_lineage=3347492,one_shot=True)
 with (H/'out/dispatch.log').open('a') as log:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(worker_pid=p.pid,queue=str(Q.relative_to(R)),log=str((H/'out/dispatch.log').relative_to(R))),indent=2))
if __name__=='__main__':
 if sys.argv[1:] == ['--start']:start()
 elif sys.argv[1:] == ['--worker']:
  try:work()
  except Exception as e:save('FAILED; no silent alternate send',error=repr(e),traceback=traceback.format_exc());raise
 else:raise SystemExit('Use --start to queue this one-shot dependency dispatcher')
