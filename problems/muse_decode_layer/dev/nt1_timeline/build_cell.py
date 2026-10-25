"""Write smallm_var_cell.py: the flag-templated NT1 kernel as a drop-in for S6's nt1_call(data, s4_nt1_bank, 'lop3', 2, 264)."""
import hashlib
from pathlib import Path
from build_var import var_source

DEV = Path(__file__).resolve().parent
TEMPLATE = '''"""Small-M NT1 variants (P3a), drop-in for S6's `nt1_call(data, state['s4_nt1_bank'], 'lop3', 2, 264)`.

`smallm_bind(s4_nt1_bank, flags_by_m)` (prepare: compiles, allocates one counter block per flags word)
`smallm_experts(data, s4_nt1_bank, flags)` (timed call; same data dict and bank as nt1_call)
Flags (bit OR): Z=1 SLOT=2 (M=1 only) ZOUT=4 RSQ=8 NREG=16  PDL=64 (launch as a programmatic dependent of the preceding kernel; the router already
triggers at its start) LAST=128.
flags=0 is the incumbent's math. Same recoded bank, same tile layout.
"""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

SMALLM_VAR_CUDA = {cuda!r}
SMALLM_VAR_CPP = """#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
int64_t max_smem_attr();int64_t smem_needed(int64_t,int64_t,int64_t);int64_t resident_blocks(int64_t,int64_t,int64_t);
void run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
"""
SMALLM_FLAGS_BUILT = {flags!r}
_SMALLM_MODULE = []


def smallm_module():
    if not _SMALLM_MODULE:
        tag = hashlib.sha256(SMALLM_VAR_CUDA.encode()).hexdigest()[:12]
        target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9, 0) else '-gencode=arch=compute_120,code=sm_120'
        mod = load_inline(name='muse_smallm_var_' + tag, cpp_sources=SMALLM_VAR_CPP, cuda_sources=SMALLM_VAR_CUDA,
            functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], no_implicit_headers=True,
            with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', target], verbose=False)
        mod.smem_optin = mod.max_smem_attr()
        _SMALLM_MODULE.append(mod)
    return _SMALLM_MODULE[0]


def smallm_bind(state, flags_by_m):
    """Prepare-time: compile and give every flags word its own zeroed counter block (64 int32)."""
    mod = smallm_module(); dev = state['gs13'].device
    for m, flags in flags_by_m.items():
        if flags not in SMALLM_FLAGS_BUILT: raise ValueError(f'flags {{flags}} not instantiated')
        if flags & 2 and m != 1: raise ValueError('SLOT is the M=1 dispatch')
        state.setdefault('smallm_counters', {{}}).setdefault(flags, torch.zeros(64, device=dev, dtype=torch.int32))
    return state


def smallm_experts(data, state, flags, grid=264):
    x = data['hidden_states'].contiguous(); m = x.shape[0]; p = m * 8; dev = x.device
    ids = data['topk_ids'].int().contiguous(); tw = data['topk_weights'].float().contiguous()
    mod = smallm_module()
    # NT2 words (2048) run 1 CTA/SM with as many stages as fit (4 on the H200); NT1 words 2 CTAs/SM, 2 stages
    stages = next((s for s in ((4, 3, 2, 1) if flags & 2048 else (2,)) if mod.smem_needed(flags, s, m) <= mod.smem_optin), 0)
    if not stages: raise ValueError('shared memory exceeds device capacity')
    grid = min(grid, state['sms'] * mod.resident_blocks(flags, stages, m))
    bank = state['banks']['lop3']
    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)
    y = torch.empty((p, 4096), device=dev); ss = torch.empty((p,), device=dev); out = torch.empty_like(x)
    tensors = [x, ids, tw, state['pnw'], bank['w13p'], bank['w2p'], state['gs13'], state['gs2'], inter, y, ss,
               state['smallm_counters'][flags], out]
    ptrs = [v.data_ptr() for v in tensors]
    mod.run(ptrs, m, data['config']['local_start'], stages, grid, torch.cuda.current_stream(dev).cuda_stream, flags)
    return out
'''


def source(keep=None):
    """keep: the flags words to instantiate (the decode lane's per-M choice); None keeps every non-trace word."""
    import re
    cuda = var_source()
    a = cuda.index('#define VAR_FLAGS'); b = cuda.index('}  // namespace', a)
    built = sorted({int(f) for f in re.findall(r'X\((\d+)\)', cuda[a:b])})
    flags = sorted(set(keep)) if keep else [f for f in built if f < 256]
    assert set(flags) <= set(built), set(flags) - set(built)
    cuda = cuda[:a] + '#define VAR_FLAGS(X) ' + ' '.join(f'X({f})' for f in flags) + '\n' + cuda[cuda.index('}  // namespace', a):]
    return TEMPLATE.format(cuda=cuda, flags=flags)


def build(keep=None):
    body = source(keep)
    out = DEV / 'smallm_var_cell.py'; out.write_text(body)
    print(out.name, hashlib.sha256(body.encode()).hexdigest())


if __name__ == '__main__':
    import sys
    build([int(x) for x in sys.argv[1:]] or None)
