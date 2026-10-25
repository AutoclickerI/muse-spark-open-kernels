"""Assemble var.cu: the admitted NT1 helpers (inc.cu, unchanged) + the flag-templated kernel."""
from pathlib import Path
DEV = Path(__file__).resolve().parent


def var_source():
    inc = (DEV / 'inc.cu').read_text()
    head = inc[:inc.index('template <int NT>\n__global__')]
    old = '  PrefetchState pre;\n};'
    assert old in head
    head = head.replace(old, '  PrefetchState pre;\n  int erank[8];  // F_SLOT: local expert of each local rank\n  int P;         // F_SLOT: local pairs\n'
                        '  int sa[4], sn[4];  // F_RND: first unit and count of each (pass, round) run of this CTA\n'
                        '  int tcnt[8];       // F_ROT: per reduce buffer, warp arrivals (monotonic)\n};')
    old = '  volatile int e, nb, m, mb, me;'
    assert old in head
    head = head.replace(old, old + '\n  volatile int seg, left;  // F_RND: current run and units left in it')
    head = head.replace('#include <vector>', '#include <vector>\n#include <algorithm>', 1)
    assert '  int wc[NCW][NLOC];' in head
    head = head.replace('  int wc[NCW][NLOC];', '  int wc[16][NLOC];  // up to 16 consumer warps (F_W16)')
    return head + (DEV / 'var_kernel.cu.inc').read_text()


if __name__ == '__main__':
    (DEV / 'var.cu').write_text(var_source())
    print('var.cu', len(var_source()))
