#!/usr/bin/env python3
"""Rebuild the standalone library-reference supplement from the common MoE lab."""
import hashlib
from build_moe_real_routing import HERE,build

if __name__=='__main__':
    build()
    path=HERE/'moe_library_refs.py'
    path.write_text((HERE/'moe_real_routing.py').read_text()+'\n'+(HERE/'moe_library_refs_driver.py').read_text())
    print(path,hashlib.sha256(path.read_bytes()).hexdigest())
