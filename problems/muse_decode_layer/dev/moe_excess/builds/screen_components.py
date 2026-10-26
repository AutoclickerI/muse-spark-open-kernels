"""Real expert arithmetic, original branch ABI, eager/capture and nearby tails."""
import importlib.util
import json
import sys
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('p3b_builder', H / 'candidates.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
spec = importlib.util.spec_from_file_location('p3b_ladder', H.parent / 'out/ladder.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

base = lab.module('night_component_BASE', (H / 'out/base.py').read_text())
checker = lab.module('night_component_TASK', lab.B['checker'])
layer = lab.module('night_component_layer', lab.B['layer'])
weights = layer.generate_weights(621713233, device='cuda', router_seed=621713237)
weights['post_norm_w'] = weights['post_expert_norm']
state = lab.prepare_moe(base, weights)
modules = {name: lab.module('night_component_' + name, (H / 'out' / (name + '.py')).read_text())
           for name in build.SPECS}
checks = []
resources = {}
hist = []
for M in (33, 128, 256, 257):
    # Nearby M33/257 exercise fallback dispatch and partial groups. The explicit
    # expert mix exercises NT16/32/48/64 and repeated 64-row chunks at both Ms.
    if M in (128, 256):
        data = lab.fixture(layer, weights, M, 621713239)
    else:
        data = checker.generate_input(bs=M, sharding='ep8', seed=621713239, device='cuda')
    fixtures = [('upstream', data)]
    if M in (128, 256):
        special = dict(data)
        ids = torch.full((M, 8), -1, device='cuda', dtype=torch.int32)
        for expert, n in enumerate((17, 31, 33, 48, 49, 65, min(M, 97), M)):
            ids[:n, expert] = expert
        special['topk_ids'] = ids
        special['topk_weights'] = torch.full((M, 8), .125, device='cuda')
        fixtures.append(('all_tail_widths', special))
    for label, d in fixtures:
        counts = [(d['topk_ids'] == e).sum().item() for e in range(32)]
        hist.append(dict(M=M, fixture=label, tokens_per_expert=counts))
        truth = checker.ref_kernel(d, weights)
        for name, module in modules.items():
            plan = module.p3b_plan(d, state['heavy'], state['native'], state['streams'])
            def call():
                for fn in plan['steps']:
                    out = fn()
                return out
            output = call()
            torch.cuda.synchronize()
            ok, message = checker.compare(output, truth, .015)
            assert ok, (M, label, name, 'eager', message)
            graph, output = lab.capture(call)
            graph.replay()
            torch.cuda.synchronize()
            ok, message = checker.compare(output, truth, .015)
            assert ok, (M, label, name, 'graph', message)
            h13, h2 = plan['owners'][:2]
            ptrs = lambda values: [v.data_ptr() for v in values]
            resources[name] = [module.p3b_night_module().night_run(ptrs(v), first,
                                module.P3B_NIGHT_KIND, 1, M, 0, False)
                               for v, first in ((h13, 1), (h2, 0))]
            checks.append(dict(M=M, fixture=label, arm=name, eager=True, graph=True))
            del graph, output, plan
            print('COMPONENT', M, label, name, 'PASS', flush=True)
(H / 'component_validation.json').write_text(json.dumps(dict(passed=True, checks=checks,
    TASK_count=2 * len(checks), resources_sm120=resources, work=hist,
    local_only=True, limitation='mma.sync stand-in; Hopper WGMMA and native occupancy need H200'), indent=2) + '\n')
