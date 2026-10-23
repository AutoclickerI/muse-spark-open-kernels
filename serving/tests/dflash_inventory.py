"""Source-derived TP8 DFlash2 arithmetic, no measured H200 latency implied."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
cfg=json.loads((ROOT/'serving/harness/hf_configs/dflash2/config.json').read_text())
H=cfg['hidden_size'];D=cfg['head_dim'];TP=8;LAYERS=cfg['num_hidden_layers']
Q=cfg['num_attention_heads']*D//TP;KV=cfg['num_key_value_heads']*D//TP
I=cfg['intermediate_size']//TP;R=cfg['dflash_config']['selector_rank'];K=16;S=7
ROWS=[]
def add(name,kind,count,shape,byte_fn,flop_fn=lambda c:0,stock='',note=''):
    cases=[]
    for c in (1,4,16,32):
        b=count*byte_fn(c);f=count*flop_fn(c)
        floor=max(b/4.8e12,f/989e12)*1e6
        cases.append(dict(c=c,bytes=b,flops=f,shape=shape(c),floor_us=floor))
    ROWS.append(dict(op=name,kind=kind,count=count,stock=stock,note=note,cases=cases))
def gemm(name,n,k,count=1,mult=8,stock='F.linear / cuBLAS',note=''):
    add(name,'dense GEMM',count,lambda c:f'[{mult*c},{k}] x [{n},{k}]T',
        lambda c:2*(n*k+mult*c*(k+n)),lambda c:2*mult*c*n*k,stock,note)
gemm('gate/up',2*I,H,LAYERS)
gemm('down',H,I,LAYERS)
gemm('conv coefficients',2*2*(H//16),H,2*LAYERS,note='Replicated, not TP-sharded; omitted by old model')
gemm('draft candidate head',cfg['vocab_size']//TP,H,mult=7,note='7 prediction slots, not 8; borrowed target BF16 head')
gemm('context FC, current',H//TP,5*H,note='Already TP8 column-split in current patch; gather below')
gemm('context FC, stock',H,5*H,note='Replicated in a55ab7d5; closed lever, not a new saving')
gemm('QKV',Q+2*KV,H,LAYERS)
gemm('o',H,Q,LAYERS)
gemm('stacked context KV',2*LAYERS*KV,H,stock='torch.mm with pretransposed NN weight',note='FusedKVMaterializeHelper already stacks five layers')
add('selector hidden','dense GEMM',1,lambda c:f'[{c},7,{H}] x [{R},{H}]T (strided bmm)',
    lambda c:2*(R*H+7*c*(H+R)),lambda c:2*7*c*R*H,
    stock='nn.Linear -> strided bmm at C>1',note='No packing clone; flattening changes BF16 rounding, keep stock')
add('grouped conv2','glue',4*LAYERS,lambda c:f'[{8*c},{H}], delta [{8*c},2,{H//16}]',
    lambda c:4*8*c*H+4*8*c*(H//16)+4*H,stock='Inductor _grouped_conv',
    note='10 pre + 10 post; BF16 inputs/output, FP32 contracted intermediates')
add('residual + RMSNorm','glue',2*LAYERS,lambda c:f'[{8*c},{H}]',
    lambda c:8*8*c*H+2*H,stock='sgl_kernel fused_add_rmsnorm',note='Includes final norm; first layer has no residual')
add('plain RMSNorm','glue',2,lambda c:f'[{8*c},{H}]',lambda c:4*8*c*H+2*H,
    stock='sgl_kernel rmsnorm',note='First input norm and context hidden_norm')
add('QK norm + RoPE','glue',LAYERS,lambda c:f'[{8*c},9,128]',
    lambda c:4*8*c*(Q+KV)+2*8*c*D+4*D+8*8*c,stock='table_qk_norm_rope_ Triton')
add('SwiGLU','glue',LAYERS,lambda c:f'[{8*c},{2*I}] -> [{8*c},{I}]',
    lambda c:6*8*c*I,stock='SiluAndMul CUDA')
add('window attention','dense GEMM (attention)',LAYERS,lambda c:f'Q[{c},8,8,128], KV[{c},2048,1,128]',
    lambda c:2*c*2048*KV*2+4*8*c*Q,
    lambda c:4*8*c*2048*Q,stock='FA3/RadixAttention BF16, noncausal block',
    note='BF16 draft-cache auto model; FP8 runtime halves KV bytes. Metadata/SFU/launch excluded')
add('selector gather + product','glue',1,lambda c:f'[{c},7,16,256] x two codebooks',
    lambda c:(209*R*2+7*R*2+7*16*8+8+2*7*16*R*2)*c,
    stock='Inductor _score_edges gathers; predecessor*hidden BF16',note='Distinct-row byte lower model; anchor reused 16 times')
add('selector edge bmm','dense GEMM',1,lambda c:f'[{7*c},16,256] x [{7*c},256,16]',
    lambda c:(2*7*16*R*2+7*16*16*2)*c,
    lambda c:2*7*c*16*16*R,stock='torch bmm / cuBLAS')
add('edge unary add','glue',1,lambda c:f'[{c},7,16,16]',lambda c:(7*16*16*6+7*16*4)*c,
    stock='Inductor _score_edges output add')
add('candidate local top16','glue',1,lambda c:f'[{7*c},25256]',
    lambda c:7*c*(25256*2+16*10),stock='FlashInfer radix top_k deterministic=True')
add('unary scale/softcap','glue',1,lambda c:f'[{7*c},16]',lambda c:7*c*16*32,
    stock='FP32 mul, div, tanh, mul (4 eager launches)',note='TP8 global topk already FP32: no cast kernel')
add('global top16 + ID gather','glue',1,lambda c:f'[{7*c},128] -> [{7*c},16]',
    lambda c:7*c*(128*4+16*4+16*8*2),stock='torch.topk + torch.gather',note='After TP candidate collective')
add('selector walk','glue',1,lambda c:f'[{c},7,16,16] selected rows -> [{c},7]',
    lambda c:1041*c,stock='selector_walk_triton',note='Already fused one program/request; not re-optimized here')
add('block IDs/positions/cacheloc','glue',1,lambda c:f'[{c},8]',lambda c:276*c,
    stock='_prepare_dflash_draft_block_contig_kernel',note='Req-map setup/compaction below may add launches')
add('accept + bonus + commit','glue',1,lambda c:f'[{c},8]',lambda c:220*c,
    stock='_dflash_accept_bonus_contig_kernel',note='Greedy path; sampled verify uses chain verifier')
add('vocab-shard embedding','glue',1,lambda c:f'[{8*c},{H}]',
    lambda c:2*8*c*H+8*8*c,stock='VocabParallelEmbedding local gather/mask; fused Triton in current patch',
    note='Add2H*U bytes for U unique owned ids (0..C+1); default no DP attention, embedding AR remains')
add('target tap packing','glue',1,lambda c:f'5x[{8*c},{H}] -> [{8*c},{5*H}]',
    lambda c:4*8*c*5*H,stock='pack_aux_hidden_states / torch.cat',note='Upstream target boundary, not a drafter matmul')
add('prefix-valid KV norm/RoPE/store','glue',1,lambda c:f'[J,{LAYERS},2,1,128], J in [{c},{8*c}]',
    lambda c:(4*LAYERS*KV+2*D+8)*c+2*LAYERS*D,
    stock='fused_kv_materialize Triton',note='Minimum committed J=c; actual bytes up to8x; scales/cache dtype preserved')
add('request map/length bookkeeping','glue',1,lambda c:f'[{c},2048] int cache map',
    lambda c:16*c*2048,stock='cache_locs rebuild_compact_draft_req_to_token_func + length ops',
    note='Window/page policy conditional; total launches need actual serving trace')
def collective(name,count,localfn,ag=False,note=''):
    cases=[]
    for c in (1,4,16,32):
        payload=localfn(c);wire=7*payload if ag else 7/8*payload
        cases.append(dict(c=c,bytes=payload*count,wire_bytes=wire*count,shape=f'{payload} B/rank',
                          floor_us=count*(2.4+wire/450e9*1e6)))
    ROWS.append(dict(op=name,kind='collective',count=count,stock='TP8 custom AR / NCCL',note=note,cases=cases))
collective('o/down all-reduces',2*LAYERS,lambda c:2*8*c*H,note='After o and down, retained')
collective('context FC all-gather',1,lambda c:2*8*c*(H//TP),True,note='Current shard; closed lever, retained')
collective('embedding all-reduce',1,lambda c:2*8*c*H,note='Full embedding cache is DP-attention-only; judge default retains this AR')
collective('candidate values all-gather, stock',1,lambda c:7*c*16*4,True)
collective('candidate IDs all-gather, stock',1,lambda c:7*c*16*8,True)
current=[]
for c in (1,4,16,32):
    size=7*c*16*(16 if c<=4 else 12);launches=1 if c<=4 else 2
    current.append(dict(c=c,bytes=size,wire_bytes=7*size,shape=f'{launches} gather(s), {size} B/rank',floor_us=launches*2.4+7*size/450e9*1e6))
ROWS.append(dict(op='candidate gather, current',kind='collective',count='1 or 2',stock='packed pairs at <=32 rows; typed otherwise',note='Existing serving port, not new work',cases=current))
report=dict(kind='source/roofline model, not H200 timings',config_sha256=hashlib.sha256((ROOT/'serving/harness/hf_configs/dflash2/config.json').read_bytes()).hexdigest(),rows=ROWS)
out=ROOT/'runs/screen/serving_dflash/inventory.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n')
header='| Op (calls/step) | Class / stock entry | Per-rank shapes c=1 / 4 / 16 / 32 | MB/step at those rungs | H200 floor us/step |\n|---|---|---|---|---|\n'
lines=[]
for r in ROWS:
    cases=r['cases']
    lines.append('| '+r['op']+f" ({r['count']}) | "+r['kind']+'; '+r['stock']+' | '+' / '.join(q['shape'] for q in cases)+' | '+' / '.join(f"{q['bytes']/1e6:.3f}" for q in cases)+' | '+' / '.join(f"{q['floor_us']:.3f}" for q in cases)+' |')
print(header+'\n'.join(lines))
