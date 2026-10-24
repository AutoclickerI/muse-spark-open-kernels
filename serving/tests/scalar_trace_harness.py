"""A single warm prefill with scalar-copy call-site attribution; not a gate."""
from pathlib import Path
import integration_harness
import run
original = run.launch


def launch(args, tree, out, port):
    args.env += [f'PYTHONPATH={tree / "python"}:{integration_harness.ROOT / "serving/tests/scalar_trace_hook"}',
                 f'MUSE_SCALAR_TRACE={out / "scalar_calls.jsonl"}']
    return original(args, tree, out, port)


run.launch = launch
if __name__ == '__main__':
    run.main()
