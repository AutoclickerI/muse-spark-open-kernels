"""Native V9 router-seed lab, off the decode quota."""
from pathlib import Path
import hashlib
from build_moe_seed_dropin import ROOT,HERE,build
from build_moe_v9_reuse import reference_generator,literal


def main():
    block=build()
    old=(HERE/'moe_v9_reuse_v2.py').read_text();pack=literal(old,'CAPTURED_R0')
    out=HERE/'moe_v9_seed.py'
    out.write_text(block+'\n'+reference_generator()+f'\nCAPTURED_R0={pack!r}\n'+(HERE/'moe_seed_lab_driver.py.inc').read_text())
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest())


if __name__=='__main__':main()
