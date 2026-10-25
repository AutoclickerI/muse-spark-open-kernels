"""Run the candidate's import-time H200 lab on sm_120 (FA3 replaced by vq8 as the q_len-8 base) to
exercise the lab code itself. Usage: python lab_local.py <candidate.py> [--break]"""
import sys
import torch
sys.path.insert(0, '/home/jihoon/competitions/muse_kernel/tools')
from local_eval import load_module  # noqa: E402
m = load_module('cand', sys.argv[1])
m._fa3_verify = lambda d: m._vq8(d)
if '--break' in sys.argv:  # force the gate's failure path once
    orig = m._vq8
    m._vq8 = lambda d, **kw: orig(d, **kw) * (1.0 + 0.5 * (torch.arange(d["config"]["num_q_heads"] * 64, device=d["qkv"].device) // 64 == 3).to(torch.bfloat16))
m._vq8_lab()
print('PLAN', m._VQ8_PLAN, 'OK', m._VQ8_OK)
