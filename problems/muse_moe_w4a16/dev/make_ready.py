from pathlib import Path
import sys
p=Path(__file__).resolve().parent.parent
is_v9 = len(sys.argv)>1 and sys.argv[1]=='v9'
s=(p/('cand_v9.py' if is_v9 else 'cand_v6.py')).read_text()
s=s.replace('NCNT = 64', 'NCNT = 20608')
s=s.replace('enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };',
'''constexpr int MAX_NB = 512;
constexpr int MAX_PAIRS = 4096;
enum { C_DONE = 0, C_OUT = NLOC * MAX_NB, C_ZERO = C_OUT + MAX_PAIRS, C_BAR, C_GEN };''')
s=s.replace('muse_moe_w4a16_v6_', 'muse_moe_w4a16_ready_')
s=s.replace('atomicAdd(cnt + C_DONE + cur_e, seg_tiles)',
            'atomicAdd(cnt + C_DONE + cur_e * MAX_NB + cur_nb, seg_tiles)')
s=s.replace('const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target)',
            'const int target = NT1;\n          while (ld_acquire(cnt + C_DONE + e * MAX_NB + nb) < target)')
s=s.replace('for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;',
'''for (int e = 0; e < NLOC; ++e)
        for (int b = 0; b < rt->nblk[e]; ++b) cnt[C_DONE + e * MAX_NB + b] = 0;''')
s=s.replace('    M = x.shape[0]\n', '    M = x.shape[0]\n    assert M <= 512\n')
s=s.replace('    nt = 1\n', '    nt = 1\n')
(p/('cand_ready9.py' if is_v9 else 'cand_ready.py')).write_text(s)
if is_v9: raise SystemExit(0)

# A separate experiment also retires token-local norm/combines before the grid reset.
z=s
needle='''        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;'''
flush='''        if (cur_pass == 1 && seg_tiles > 0) {
          __threadfence();
          bar_consumers();
          if (tid < NB) {
            const int idx = cur_nb * NB + tid;
            if (idx < rt->n[cur_e]) atomicAdd(cnt + C_OUT + rt->off[cur_e] + idx, seg_tiles);
          }
        }
'''
z=z.replace(needle, flush+needle)
start=z.index('  // ---------------- grid barrier ----------------')
end=z.index('  // ---------------- combine:',start)
bar=z[start:end]
z=z[:start]+flush.replace('        ','  ')+z[end:]
# Move reset after all token-local consumers. C_OUT is published after y and ss writes.
needle='''      if (pos >= 0) {
        const float sc = tw[t * 8 + j]'''
z=z.replace(needle, '''      if (pos >= 0) {
        while (ld_acquire(cnt + C_OUT + pos) < NT2) __nanosleep(64);
        const float sc = tw[t * 8 + j]''')
bar=bar.replace('    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {',
'''    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {
      const int P = rt->off[NLOC - 1] + rt->n[NLOC - 1];
      for (int i = 0; i < P; ++i) cnt[C_OUT + i] = 0;''')
# End of CUDA device function is immediately before the host wrapper's smem helper.
marker='\n}\n\nsize_t smem_bytes('
assert marker in z
z=z.replace(marker, '\n'+bar+'}\n\nsize_t smem_bytes(',1)
z=z.replace('muse_moe_w4a16_ready_', 'muse_moe_w4a16_retire_')
(p/'cand_retire.py').write_text(z)
