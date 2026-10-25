"""Assemble var.cu: the admitted NT1 helpers (inc.cu, unchanged) + the flag-templated kernel."""
from pathlib import Path
DEV = Path(__file__).resolve().parent


def var_source():
    inc = (DEV / 'inc.cu').read_text()
    head = inc[:inc.index('template <int NT>\n__global__')]
    old = '  PrefetchState pre;\n};'
    assert old in head
    head = head.replace(old, '  PrefetchState pre;\n  int erank[8];  // F_SLOT: local expert of each local rank\n  int P;         // F_SLOT: local pairs\n  int uq[8];     // F_DYN: unit queued in each stage (-1 = end)\n  int gnext, gend, dk;  // F_DYN: grabbed chunk [gnext, gend) and chunk size\n};')
    head = head.replace('#include <vector>', '#include <vector>\n#include <algorithm>', 1)
    return head + (DEV / 'var_kernel.cu.inc').read_text()


if __name__ == '__main__':
    (DEV / 'var.cu').write_text(var_source())
    print('var.cu', len(var_source()))
