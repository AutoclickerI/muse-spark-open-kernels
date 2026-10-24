"""Rebuild the archived, inadmissible BF16-resident experiment for evidence only."""
from pathlib import Path
from build_lab import prefix,ROOT,DECODE

BASE='cand_r3_v9.py'


def main():
    path=Path(__file__).parent
    fixture=(DECODE/'moe_tune.py').read_text()
    fixture=fixture[fixture.index('HIDDEN = 8192'):fixture.index('def _graph_time')]
    source=(DECODE/BASE).read_text().rstrip()
    source+='\n\n# BF16-resident hybrid: exact block products + real-router prepare selection.\n'
    source+=prefix(fixture,'_bf_fixture_')+'\n'
    source+=(path/'hybrid.py').read_text()+'\n'+(path/'prepare_tail.py').read_text()
    out=DECODE/'dev/parked/cand_r3_bf16moe.py'
    out.write_text(source)
    import hashlib
    print(out,len(source),hashlib.sha256(source.encode()).hexdigest())

if __name__=='__main__':main()
