"""Decode every pipe probe from a local log or h200_lab receipt JSON without truncation."""
import argparse
import base64
import json
import lzma
from pathlib import Path


def unpack(text):
    return [lzma.decompress(base64.b85decode(line.split(' ', 1)[1])).decode()
            for line in text.splitlines() if line.startswith('PIPE_XZ85 ')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt', type=Path)
    a = p.parse_args()
    text = a.receipt.read_text()
    if a.receipt.suffix == '.json':
        text = json.loads(text)['result']['log_tail']
    bodies = unpack(text)
    assert len(bodies) == 1, f'expected one complete PIPE_XZ85 record, got {len(bodies)}'
    print(bodies[0])


if __name__ == '__main__':
    main()
