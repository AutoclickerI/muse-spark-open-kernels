"""Small, recorded same-server sampling diagnostic; no changes to model arithmetic."""
import argparse
import json
import time
import urllib.request

import common
import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='sampling-diagnostic')
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('extra', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    out = common.RESULTS / args.label
    out.mkdir(parents=True, exist_ok=False)
    target, draft = common.model_dirs('random')
    opts = argparse.Namespace(target=str(target), draft=str(draft), tp=1,
        mem_fraction_static=.85, max_total_tokens=300000,
        attention_backend=None, extra=[x for x in args.extra if x != '--'], env=args.env, gpus='1')
    proc, cmd = run.launch(opts, common.STOCK_TREE, out, run_port := run.free_port())
    records = []
    try:
        run.wait_ready(proc, run_port, out / 'server.log', 300)
        prompt = run.build_workload(0, [1], {1: 1}, 1, [2600])[0]['input_ids']
        for params in [dict(temperature=0.0), dict(temperature=0.0),
                       dict(temperature=0.0, top_k=1), dict(temperature=0.0, top_k=1)]:
            payload = dict(input_ids=prompt, sampling_params=dict(
                **params, max_new_tokens=8, ignore_eos=True),
                return_logprob=True, top_logprobs_num=5)
            req = urllib.request.Request(f'http://127.0.0.1:{run_port}/generate',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            t = time.monotonic()
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.load(resp)
            records.append(dict(params=params, result=result, seconds=time.monotonic()-t))
            (out / 'responses.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(records[-1]), flush=True)
    finally:
        run.stop(proc)
        (out / 'command.json').write_text(json.dumps(cmd, indent=2))


if __name__ == '__main__':
    main()
