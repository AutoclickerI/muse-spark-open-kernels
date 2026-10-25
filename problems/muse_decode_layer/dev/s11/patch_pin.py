"""Pin the prepare tuners' choices to whole-layer paired winners (dev/s11/pins_lab.py.inc) instead of per-run timing.

The projection tuner times each role alone (3 graph repeats) and flips between runs: S8 / S9 / S10 picked
verify-1 qkv ('lt', 2) / 'custom' / 'cublas', decode-16 o 'cublas' / 'cublas' / ('lt', 0), decode-16 unpooler
('lt', 1) / 'custom' / 'cublas'; the attention tuner flipped decode-16/32 sliding between 'layer' and 'attn_main'.
PROJ_PINS (tokens, role) -> choice: _tune_projections still builds the row's inputs (cuBLASLt algos included) but times
only the pinned variant (a short_lt pin times 'custom' there; _tune_measured_lt then builds its layout, runs its 1% FP32
guard and sets it); _tune_measured_lt never overrides another pinned row. ATTN_PINS (batch, q_len, window) -> choice
goes into BEST7_SETTINGS' attention column, gated by _best7_gate_attention like every fixed row.
"""
import ast
import re

# Whole-layer one-factor sweep on S11 cf493665 (4bea7724, 12 seeds x 6 rounds, cold L2; r = LOAD ratio vs that run's
# tuner pick). t <= 32: custom (balanced / V9 tile) beats every cuBLAS / cuBLASLt variant in the layer; the run's
# tuner picked cuBLAS for o at t = 8 / 16 / 32 (r 0.981 / 0.986 / 0.994), qkv and unpooler at t = 16 (0.984, 0.992).
# t = 128 / 256: every variant within 0.5% of the next; the sweep's best.
PROJ = {**{(t, r): 'custom' for t in (1, 8, 16, 32) for r in ('qkv_gate', 'pool_router')},
        (1, 'unpooler_slice'): 'custom', (8, 'unpooler_slice'): 'custom', (16, 'unpooler'): 'custom', (32, 'unpooler'): 'custom',
        (8, 'o'): 'custom', (16, 'o'): 'custom', (32, 'o'): 'custom',
        (128, 'qkv_gate'): ('vdsk', (128, 64, 64, 8, 4, 1), 3), (128, 'o'): ('lt', 0), (128, 'pool_router'): 'cublas',
        (128, 'unpooler'): ('short_lt', 1, 0),
        (256, 'qkv_gate'): ('lt', 0), (256, 'o'): 'custom', (256, 'pool_router'): 'custom', (256, 'unpooler'): ('short_lt', 0, 0)}
# Attention at decode 16 / 32 (BEST7_SETTINGS 'auto'): attn_main wins all four in 4bea7724; 'layer' costs +0.6 / +6.5 /
# +0.25 / +8.4% (d16S / d16F / d32S / d32F), FA3 +6 to +56%.
ATTN = {(16, 1, 2048): ('attn_main', None), (16, 1, 0): ('attn_main', None), (32, 1, 2048): ('attn_main', None),
        (32, 1, 0): ('attn_main', None)}

EDITS = (
    ('            scores=[]\n            want=None\n',
     '            if (t, name) in PROJ_PINS:\n'
     '                pin = PROJ_PINS[(t, name)]\n'
     "                variants = ['custom' if isinstance(pin, tuple) and pin[0] == 'short_lt' else pin]\n"
     '            scores=[]\n            want=None\n'),
    ('            if name == "pool_router" and t <= 32:\n                plan[(t, name)] = "custom"\n',
     '            if name == "pool_router" and t <= 32 and PROJ_PINS.get((t, name), "custom") == "custom":\n'
     '                plan[(t, name)] = "custom"\n'),
    ("            if ratio<1.0:state['proj_plan'][(t,name)]=choice\n",
     "            if (ratio<1.0 if (t,name) not in PROJ_PINS else PROJ_PINS[(t,name)]==choice):state['proj_plan'][(t,name)]=choice\n"),
)
ANCHOR = 'def _tune_projections(state, flush, device):\n'


def _settings_pin(src, attn):
    """Rewrite the attention column of the BEST7_SETTINGS rows named in attn (the dict literal on one line)."""
    m = re.search(r'^BEST7_SETTINGS = (\{.*\})$', src, re.M)
    table = ast.literal_eval(m.group(1))
    for key, choice in attn.items():
        row = list(table[key]);assert row[1] == 'auto', (key, row[1]);row[1] = choice;table[key] = tuple(row)
    return src[:m.start(1)] + repr(table) + src[m.end(1):]


def patch(src, proj=None, attn=None):
    proj = PROJ if proj is None else proj
    attn = ATTN if attn is None else attn
    for old, new in EDITS:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new)
    assert src.count(ANCHOR) == 1
    src = src.replace(ANCHOR, '# Whole-layer paired winners (dev/s11/patch_pin.py): (tokens, role) -> projection choice\n'
                      f'PROJ_PINS = {proj!r}\n\n\n' + ANCHOR)
    return _settings_pin(src, attn) if attn else src
