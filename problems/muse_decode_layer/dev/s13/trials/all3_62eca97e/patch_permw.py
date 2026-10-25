"""PERMW: wide packed banks + scale-bit rotation + hoisted descriptors (b88d04fe)."""
import importlib.util
from pathlib import Path

def patch(src):
    for name in ('wide_frozen', 'permw_frozen'):
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        src=m.patch(src)
    return src
