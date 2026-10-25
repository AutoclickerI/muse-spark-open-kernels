"""Normal / forced local checks of a decode candidate carrying the coded patch: the integration lane's
problems/muse_decode_layer/dev/s9/audit.py (local_eval tests + loads; --forced forces the Hopper glue through
best7/force.py), plus a count of coded GEMV calls by (tokens, N, S), eager and captured, to show which rows run coded.

    CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/coded_dense/check_cand.py <cand.py> <out_dir> [--forced]
"""
import importlib.util
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R / 'tools'))
import local_eval  # noqa: E402


def main():
    cand, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    forced = '--forced' in sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('s9_audit', R / 'problems/muse_decode_layer/dev/s9/audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    counts = {}
    original = local_eval.load_module

    def loader(name, path):
        m = original(name, path)
        if name == 'submission':
            gemv = m.cd_gemv

            def counted(x, c, S, **kw):
                row = counts.setdefault(f"T{x.shape[0]}:N{c['N']}:K{c['K']}:S{S}", dict(eager=0, captured=0))
                row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager'] += 1
                return gemv(x, c, S, **kw)
            m.cd_gemv = counted
        return m
    local_eval.load_module = loader
    sys.argv = ['audit.py', '--candidate', str(cand), '--out', str(out)] + (['--forced'] if forced else [])
    try:
        audit.main()
    finally:
        (out / ('coded_calls_forced.json' if forced else 'coded_calls.json')).write_text(json.dumps(counts, indent=2) + '\n')
        print('CODED_CALLS', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
