"""Prepare a fresh-seed direct PAIR comparison for measured rebuild winners only."""
import argparse
import ast
import hashlib
from pathlib import Path

from build_epilogue_pair import source as pair_source
from build_pipe_ready import KINDS, source

D = Path(__file__).resolve().parent
R = D.parents[3]


def build(arms, output):
    driver = (D / 'pipe_rebuild_lab.py.inc').read_text()

    def replace(before, after):
        nonlocal driver
        assert driver.count(before) == 1, before
        driver = driver.replace(before, after)

    selected = {arm: KINDS[arm.lower()] for arm in arms}
    replace("REBUILD_ARMS={'FLIGHT':'pfly','ILP':'pilp','K16':'pk16','LUT':'plut'}",
            'REBUILD_ARMS=' + repr(selected))
    replace("out=smallm_experts(d,st,69) if arm=='S14' else globals()[REBUILD_ARMS[arm]+'_experts'](d,st,69)",
            "out=(epit_experts(d,st,69) if arm=='PAIR' else smallm_experts(d,st,69) if arm=='S14' "
            "else globals()[REBUILD_ARMS[arm]+'_experts'](d,st,69))")
    start = driver.index(" lines=['META P3a conditional rebuild")
    end = driver.index('\n fns=', start)
    driver = driver[:start] + (
        " lines=['META P3a fresh-seed direct comparison: exact H200-nominated rebuild cells versus PAIR, plus S14 back. "
        "Same packed bank/full work/precision. No assumption that PAIR and rebuild gains add.',\n"
        " 'META 32 new weight/router seeds at M8/16/32; PAIR control, ABBAx6 readflush, A:A mean/max, "
        "before/after native TASK gates and consumer nodes. No whole-layer promotion from this stage test.']"
    ) + driver[end:]
    replace('fns=[smallm_module,activity_module]', 'fns=[smallm_module,epit_module,activity_module]')
    replace('5353+7919*seed', '15353+7919*seed')
    replace('9000+m+97*seed', '19000+m+97*seed')
    replace("smallm_bind(st,{8:69})", "smallm_bind(st,{8:69});epit_bind(st,{8:69})")
    replace("for arm in ('S14',*REBUILD_ARMS):", "for arm in ('PAIR','S14',*REBUILD_ARMS):")
    replace("if 'S14' not in graphs:continue", "if 'PAIR' not in graphs or 'S14' not in graphs:continue")
    replace("local=graphs['S14'].replay()[4]", "local=graphs['PAIR'].replay()[4]")
    replace("ctrl=TimelineGraph(lambda:rebuild_routed(pr,rs,st,'S14'));a,b,_=timeline_timing(graphs['S14'],ctrl)",
            "ctrl=TimelineGraph(lambda:rebuild_routed(pr,rs,st,'PAIR'));a,b,_=timeline_timing(graphs['PAIR'],ctrl)")
    replace('order=list(REBUILD_ARMS)', "order=['S14',*REBUILD_ARMS]")
    replace("timeline_timing(graphs['S14'],graphs[arm])", "timeline_timing(graphs['PAIR'],graphs[arm])")
    replace('S14_us=', 'PAIR_us=')
    replace("abs_us={statistics.mean(map(abs,vs)):.3f}",
            "abs_us={statistics.mean(map(abs,vs)):.3f} max_abs_us={max(map(abs,vs)):.3f} n={len(vs)}")
    text = (R / 'prof/lab/moe_nt1_reuse.py').read_text() + '\n' + pair_source()[0]
    for arm in arms:
        text += '\n' + source(arm.lower())[0]
    text += '\n' + driver
    ast.parse(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(output, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs='+', choices=('FLIGHT', 'ILP', 'K16', 'LUT'))
    parser.add_argument('-o', type=Path, required=True)
    args = parser.parse_args()
    build(args.arms, args.o)
