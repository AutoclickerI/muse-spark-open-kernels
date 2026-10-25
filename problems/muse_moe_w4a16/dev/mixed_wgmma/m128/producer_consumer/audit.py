"""Audit the failed D split and the new expert-wide pipeline from native SASS."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def compile_source(p):
    cube=p.with_suffix('.cubin')
    r=subprocess.run(['nvcc','--cubin','-O3','-std=c++20','-arch=sm_90a','-lineinfo','-Xptxas=-v',str(p),'-o',str(cube)],capture_output=True,text=True)
    p.with_suffix('.ptxas.log').write_text(r.stdout+r.stderr);r.check_returncode()
    sass=subprocess.run(['cuobjdump','--dump-sass',str(cube)],capture_output=True,text=True,check=True).stdout;p.with_suffix('.sass').write_text(sass)
    lined=subprocess.run(['nvdisasm','-c','-g',str(cube)],capture_output=True,text=True,check=True).stdout;p.with_suffix('.lines.sass').write_text(lined)
    registers={};name=None
    for line in r.stderr.splitlines():
        if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
        m=re.search(r'Used (\d+) registers',line)
        if m:registers[name]=int(m[1])
    entries=[]
    for part in sass.split('Function : ')[1:]:
        name=part.splitlines()[0].strip();ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Za-z0-9_.]*)\s*(.*?)\s*;',part)
        counts=Counter(op for op,arg in ops);local=sum(op.startswith(('LDL','STL')) for op,arg in ops)
        assert local==0,(name,local)
        if 'heavyI' in name:
            mma=[arg for op,arg in ops if op.startswith('HGMMA.')];assert mma and all(re.match(r'R\d+, gdesc\[UR\d+\], R\d+',arg) for arg in mma),(name,mma[:1])
            assert all(arg.endswith('0x0') for op,arg in ops if op.startswith('WARPGROUP.DEPBAR'))
        entries.append(dict(name=name,registers=registers[name],local_memory_instructions=local,static_opcode_counts=dict(counts)))
    return dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kernels=entries,no_C7513='C7513' not in r.stderr)


def bank_gate():
    # D's actual emitted 128-bit LDS/STS requests are serviced in eight-lane
    # (128-byte) waves. A four-word lane stride visits every bank once/wave.
    result={}
    for stride in (4,16):
        waves=[]
        for first in range(0,32,8):
            visits=Counter((lane*stride+j)%32 for lane in range(first,first+8) for j in range(4))
            waves.append(max(visits.values()))
        result[str(stride)]=dict(worst_distinct_words_per_bank=max(waves),waves=waves)
    assert result['4']['worst_distinct_words_per_bank']==1
    assert result['16']['worst_distinct_words_per_bank']==4
    # New producer scatter is a bijection with one 32-bit word/lane/bank.
    written=set();max_bank=0
    for atom in range(4):
        for j in range(4):
            for w in range(4):
                visits=Counter()
                for l in range(32):
                    r=w*16+l//4+(j&1)*8;k=atom*16+(l&3)*2+(j>>1)*8;idx=r*64+(k^((r&7)*8))
                    assert idx not in written;written.add(idx);visits[(idx//2)%32]+=1
                max_bank=max(max_bank,max(visits.values()))
    assert len(written)==2048 and max_bank==1
    return dict(old_D_lane_strides=result,new_scatter_words=len(written),new_scatter_words_per_bank=max_bank)


def ring_gate():
    # Exhaust all producer/consumer interleavings for short rings. A producer
    # may prefetch S-1 future stages; it must own empty before generic/cp.async
    # writes. A consumer must finish every token subgroup before release.
    cases=[]
    for stages in (2,3):
        for steps in (stages,stages+1,2*stages+1):
            initial=(0,0,0,tuple(-1 for _ in range(stages)))
            todo=[initial];seen={initial};finished=0
            while todo:
                loaded,published,consumed,owners=todo.pop()
                assert 0<=consumed<=published<=loaded<=steps
                assert loaded-consumed<=stages
                if consumed==steps:finished+=1;continue
                moves=[]
                if loaded<steps and loaded-published<stages:
                    st=loaded%stages
                    if owners[st]==-1:
                        own=list(owners);own[st]=loaded;moves.append((loaded+1,published,consumed,tuple(own)))
                if published<loaded:moves.append((loaded,published+1,consumed,owners))
                if consumed<published:
                    st=consumed%stages;assert owners[st]==consumed
                    own=list(owners);own[st]=-1;moves.append((loaded,published,consumed+1,tuple(own)))
                assert moves,'ring deadlock'
                for move in moves:
                    if move not in seen:seen.add(move);todo.append(move)
            assert finished
            cases.append(dict(stages=stages,steps=steps,states=len(seen)))
    return cases


def main():
    payload=(ROOT/'prof/lab/moe_nt1_fixed.py').read_text()
    frozen=next((ROOT/'runs/muse_moe_w4a16').glob('*383ccc1fbbae41e2.py'))
    t=ast.parse(frozen.read_text());all_payloads=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_LAB_PAYLOADS')
    accepted=next(text for name,sha,text in all_payloads if name=='moe_nt1_fixed')
    assert hashlib.sha256(payload.encode()).digest()==hashlib.sha256(accepted.encode()).digest()
    tree=ast.parse(accepted)
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NT1_SOURCES')
    result={}
    for name in ('INC','D','D128'):
        p=HERE/('audit_'+name+'.cu');p.write_text(src[name]);result[name]=compile_source(p)
    new=compile_source(HERE/'pc.cu');assert new['no_C7513']
    result['PC']=new
    report=dict(native=result,source_matches_H200383ccc=True,accepted_payload_sha256=hashlib.sha256(accepted.encode()).hexdigest(),bank_model=bank_gate(),ring_interleavings=ring_gate(),
        old_tile_model=dict(weight_values=65536,NT1_raw_TMA_requests=1,D_raw_TMA_requests=8,D128_raw_TMA_requests=4,
            D_mbarrier_arrivals_per_tile=128,D128_mbarrier_arrivals_per_tile=64,NT1_empty_arrivals_per_tile=8,
            D_decoded_shared_write_bytes=131072,D_decoded_shared_read_bytes=131072,
            D_producer_threads=128,D_consumer_threads=256,NT1_decode_threads=256,
            D_consumer_A_loads=32,D128_consumer_A_loads=32,NT1_register_A_roundtrip_bytes=0,
            D_consumer_B_LD64_per_lane_per_tile=32,NT1_B_LD128_per_lane_per_expert_block_segment=16,
            D_consumer_B_bytes_per_lane_per_tile=256,NT1_B_bytes_per_lane_per_segment=256,
            B_register_cache_regression='NT1 retains Bf across cc contiguous channel tiles for a fixed expert/token block; D removes Bf and reloads all B for every channel tile. B load issue grows2*cc and B bytes growcc, before caching. cc=min(ntile,max(1,round(U/grid))); this is a logical request count, not DRAM bytes.',
            D_BF16x2_mul_per_producer_lane_per_microstage=32,D128_BF16x2_mul_per_producer_lane_per_microstage=64,
            D_mma_per_consumer_lane_per_microstage=4,D128_mma_per_consumer_lane_per_microstage=8,
            issue_floor='D64 has128 producer-warp BF16x2 mul instructions plus32 consumer-warp MMA issues per microstage before unpack/scale/address/store/load/barriers. At4 warp instructions/cycle this is40cycles/8192 weights across one CTA, not a measured clock rate; unpack and address work add substantially.',
            note='Logical loop counts for one original16x4096 tile/token block, excluding initialization/epilogue/poll retries. Arrival totals count all warp leaders, not warp-instruction throughput.'),
        occupancy_model=dict(SM90_registers=65536,CTA_register_rounding='8 registers/thread at warp granularity',
            INC=dict(registers=125,threads=256,rounded_CTA_registers=32768,max_CTAs_by_registers=2,consumer_warps_at_2_CTAs=16),
            D=dict(registers=70,threads=384,rounded_CTA_registers=27648,max_CTAs_by_registers=2,consumer_warps_at_2_CTAs=16,producer_warps_at_2_CTAs=8),
            D128=dict(registers=78,threads=384,rounded_CTA_registers=30720,max_CTAs_by_registers=2),
            note='Register-derived upper bounds agree with H200 receipt grid264. Grid is not an independent occupancy counter; actual active blocks are printed in the new lab.'),
        inference='No spills and no submitted-layout bank conflict. D drops register-cached activations across channel tiles, halves decode warp width, adds BF16 shared roundtrip,8/4 fine-grain TMA+handshake stages, extra A/B load and address issue, and per-microtile schedule decoding. D128 gains18% while retaining B reloads and roundtrip costs. Counts establish added work, not exact causal stall percentages.')
    (HERE/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print('PC_AUDIT_PASS',len(new['kernels']),'new kernels; old D no spills; conflict-free; ring states',sum(v['states'] for v in report['ring_interleavings']),flush=True)


if __name__=='__main__':main()
