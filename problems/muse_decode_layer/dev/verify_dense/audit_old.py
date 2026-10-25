from pathlib import Path
import ast,json,hashlib,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OLD=ROOT/'problems/muse_bf16_proj/dev/verify_streaming'
configs=ast.literal_eval(next(n.value for n in ast.parse((OLD/'projection.py').read_text()).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='VS_CONFIGS' for t in n.targets)))
source=ast.literal_eval(next(n.value for n in ast.parse((OLD/'projection.py').read_text()).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='VS_CUDA' for t in n.targets)))
old_info=json.loads((OLD/'compile_sm90a.json').read_text());assert old_info['source_sha256']==hashlib.sha256(source.encode()).hexdigest()
sass=(OLD/'sm90a.sass.log').read_text();rows=[]
log=(ROOT/'problems/muse_moe_w4a16/lab/profiles/lab_verify_streaming_synthetic_moe.19f3a13b17b44b6f.log').read_text()
for line in log.splitlines():
 if not line.startswith('M='):continue
 get=lambda k:re.search(r'\b'+k+r'=([^ ]+)',line).group(1)
 m=int(get('M'));role=get('role');n,k=dict(qkv_gate=(2304,8192),o=(8192,1024),pool_router=(4608,8192),unpooler=(8192,4096))[role]
 code=int(get('code'));bm,bn,bk,st,pipe=configs[code];tiles=((m+bm-1)//bm)*((n+bn-1)//bn);smem=st*(bm+bn)*bk*2+st*16
 marker=f'gemmILi{bm}ELi{bn}ELi{bk}ELi{st}ELb{int(pipe)}E'
 starts=[v for v in re.split(r'Function : ',sass)[1:] if marker in v.splitlines()[0]];assert len(starts)==1,(marker,len(starts))
 block=starts[0];counts={x:len(re.findall(x,block)) for x in ('HGMMA','UTMALDG','WARPGROUP','SYNCS','STG','LDG','ATOM','NANOSLEEP')}
 kernels=[v for v in old_info['functions'] if marker in v['name']];assert len(kernels)==1
 row=dict(m=m,role=role,code=code,tile=[bm,bn,bk],stages=st,pipe=pipe,ctas=tiles,sms=132,first_wave_sm_fill=min(1,tiles/132),waves=(tiles+131)//132,nominal_wave_efficiency=tiles/(132*((tiles+131)//132)),smem_bytes=smem,max_blocks_by_smem=(228*1024)//smem,k_steps=k//bk,instructions=counts,native=kernels[0]);rows.append(row)
result=dict(source_sha256=old_info['source_sha256'],sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),rows=rows,
 conclusions=['QKV winning tile64x128x64 stages6 pipe0 has only36/72 CTAs at M128/256, one resident CTA/SM by147552B shared memory. It issues128 K64 iterations and drains WGMMA at every iteration.',
 'Pool/router winners have72 CTAs at both verify loads; the M256 variant has two math warpgroups but still one CTA/SM.',
 'O and unpooler winners have128 CTAs: 97% first-wave fill. Their loss cannot be blamed on low CTA count alone: K64 requests, wait/fence density, narrow GMMA and scalar stores remain.',
 'No spills in all15 kernels. SASS contains real HGMMA and TMA, so Hopper did not silently use the portable MMA path.',
 'Regular split-K adds output partial writes, atomics and deterministic serial accumulation. It does not supply balanced Stream-K scheduling. All measured splits>1 lose.',
 'No H200 hardware-counter trace exists: fill is exact source geometry; latency attribution is an inference, not a measured stall breakdown.'])
(HERE/'audit_old.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
