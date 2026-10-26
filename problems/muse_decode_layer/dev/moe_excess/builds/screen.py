"""S22's normal/forced stand-ins, recording every new heavy launch."""
import importlib.util
import types
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parents[2]
spec = importlib.util.spec_from_file_location('p3b_s22_screen', P / 'dev/s22/screen.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def install(module, forced=False):
    records = prior.install(module, forced)
    if hasattr(module, 'p3b_night_module'):
        original = module.p3b_night_module
        def extension():
            native = original()
            def launch(vector, first, kind, grid, M, stream, run):
                if run:
                    key = 'NIGHT_' + str(kind) + ('_W13' if first else '_W2')
                    row = records.setdefault(key, dict(eager=0, captured=0))
                    row['captured' if module.torch.cuda.is_current_stream_capturing() else 'eager'] += 1
                return native.night_run(vector, first, kind, grid, M, stream, run)
            return types.SimpleNamespace(night_run=launch)
        module.p3b_night_module = extension
    return records
