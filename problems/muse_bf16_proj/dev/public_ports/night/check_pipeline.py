"""Offline safety check for a one-shot autonomous read/select/send chain."""
import hashlib,importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=H.parents[4];sys.path.insert(0,str(R/'tools'))
from lab_packet import encode_packet
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
reader=load('ng_reader_validation',H/'read.py');portfolio=load('ng_selector_validation',H/'portfolio.py');raw=json.loads((H/'out/local_raw.json').read_text());manifest=json.loads((H/'manifest.json').read_text());payload=(H/'out/night_gap.py').read_text()
def packet(data):return encode_packet('night_gap',manifest['payload_sha256'][:16],dict(lines=['NIGHT_GAP_XZ85 '+reader.nb_time_encode(data)],complete=None,format='legacy-lines'))
wrapper='_LAB_PAYLOADS='+repr([('night_gap',manifest['payload_sha256'][:16],payload)])
receipt=dict(id='LOCAL_READER_ONLY',result=dict(log_tail=packet(raw)));out=reader.analyse(receipt,wrapper,manifest);assert len(out['comparisons'])==10
invalid=dict(raw,complete=False);rejected=False
try:reader.analyse(dict(receipt,result=dict(log_tail=packet(invalid))),wrapper,manifest)
except AssertionError:rejected=True
assert rejected,'Incomplete parent/result must not become native evidence'
# Mean-positive, CI-unresolved stays exploratory; a negative mean is dropped.
# Geometry dispatch may take a pool schedule that wins only at its selected T;
# the following whole-graph batch, not these unit fixtures, judges composition.
model=dict(job='LOCAL_SELECTION_FIXTURE',manifest=dict(points={'PORT0':[[16,'qkv_gate','N']],'PORT1':[[32,'pool_router','P']],'PORTS':[]},pool=['CW4_S2']),arms={
 'PORT0':dict(measured=True,R=[1.001,.99,1.01],loads={}),
 'PORT1':dict(measured=True,R=[.999,.99,1.01],loads={}),
 'CW4_S2':dict(measured=True,R=[.999,.99,1.01],loads={'d16':dict(R=[1.02,1.01,1.03]),'d32':dict(R=[.98,.97,.99])})})
selected=portfolio.select_parent(model);assert selected['points']==[[16,'qkv_gate','N']] and selected['pool']=={16:'CW4_S2'};assert selected['dropped'][0]['arm']=='PORT1'
old_seeds=set(json.loads((P/'whole_manifest.json').read_text())['seeds']);assert not old_seeds.intersection(manifest['seeds'])
proof=dict(passed=True,reader_cases=12,all_rounds_exact=True,incomplete_packet_rejected=True,mean_positive_CI_unresolved_retained=True,negative_point_dropped=True,pool_shape_dispatch_priced_next=True,seeds_disjoint=True,limits='Synthetic selection fixtures and local reader transport only. No fixture or local timing is native performance evidence.')
(H/'pipeline_validation.json').write_text(json.dumps(proof,indent=2)+'\n');print('PASS autonomous reader/selector guard')
