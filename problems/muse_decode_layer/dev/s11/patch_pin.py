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

PROJ = {}
ATTN = {}

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
