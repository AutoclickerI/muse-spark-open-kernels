"""Diagnostic CUDA/CPU profile of the actual c32 margin workload, outside gates."""
import json
import sys
import urllib.request
from pathlib import Path
import integration_harness
import run

original = run.run_level
label = sys.argv[sys.argv.index('--label') + 1]
profile_dir = Path('/data2/jihoon/muse_kernel/harness/results') / label / 'profile'


def level(port, concurrency, items, timeout, on_result=None):
    if concurrency == 32:
        payload = dict(activities=['CPU', 'GPU'], num_steps=16,
                       output_dir=str(profile_dir), profile_by_stage=False,
                       with_stack=False, record_shapes=False, profile_prefix=label)
        req = urllib.request.Request(f'http://127.0.0.1:{port}/start_profile',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            print('PROFILE', response.read().decode(), flush=True)
    return original(port, concurrency, items, timeout, on_result=on_result)


run.run_level = level
if __name__ == '__main__':
    run.main()
