"""Freeze a serving source snapshot for provenance-checked harness comparisons."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = 'a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f'


def freeze(destination, source=ROOT / 'serving/sglang'):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    subprocess.run(['git', '-C', str(ROOT / 'serving/sglang'), 'worktree', 'add',
                    '--detach', str(destination), BASE], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['rsync', '-a', '--delete', '--exclude=__pycache__',
                    str(Path(source) / 'python') + '/', str(destination / 'python') + '/'], check=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'serving/sglang')
    args = parser.parse_args()
    print(freeze(args.destination, args.source))
