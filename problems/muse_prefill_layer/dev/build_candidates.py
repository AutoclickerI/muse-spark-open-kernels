from pathlib import Path
p=Path('problems/muse_prefill_layer')
v1=(p/'cand_v1.py').read_text()
v2=v1.replace('if IS_HOPPER:\n        _bench_w4a16(weights)', 'if False:\n        _bench_w4a16(weights)')
v2=v2.replace('H=HIDDEN, num_warps=16','H=HIDDEN, num_warps=4')
v2=v2.replace('BRS = 16','BRS = 1').replace('D=D_EXPERT, num_warps=8)\n    y =', 'D=D_EXPERT, num_warps=4)\n    y =')
# Keep all materialized expert activations at their SGLang precision after applying the global scale.
v2=v2.replace('y = g * tl.sigmoid(g) * u', 'g = g.to(tl.bfloat16).to(tl.float32)\n    u = u.to(tl.bfloat16).to(tl.float32)\n    y = g * tl.sigmoid(g) * u')
v2=v2.replace('y = tl.load(y_ptr + r.to(tl.int64) * D + c).to(tl.float32) * tl.load(gs_ptr + e) * pw', 'y = (tl.load(y_ptr + r.to(tl.int64) * D + c).to(tl.float32) * tl.load(gs_ptr + e)).to(tl.bfloat16).to(tl.float32) * pw')
(p/'cand_v2.py').write_text(v2)
host=(p/'dev/moe_host.py').read_text()
host=host.replace('src = open(src_path).read()', 'src = CUDA_SRC')
cuda=(p/'dev/moe_w4a16.cu').read_text()
v3=v2.replace('def prepare(weights):','def prepare_base(weights):')
v3 += '\n\nCUDA_SRC = '+repr(cuda)+'\n\n'+host
v3 += '''

def prepare(weights):
    st = prepare_base(weights)
    st['blob13'], st['sfb13'] = repack(weights['w13'], weights['w13_sf'], True)
    st['blob2'], st['sfb2'] = repack(weights['w2'], weights['w2_sf'], False)
    st['tiled_mod'] = load_module(None)
    st['sms'] = torch.cuda.get_device_properties(weights['w13'].device).multi_processor_count
    return st
'''
v3=v3.replace('moe = _experts_dequant(pooled, topk_w, slot_e, slot_pos, counts, st)', 'moe = experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st)')
# Exercise the new expert path on every test and load; the measured crossover will set later dispatch.
v3=v3.replace('BIG_TOKENS = 2048', 'BIG_TOKENS = 0')
(p/'cand_v3.py').write_text(v3)
