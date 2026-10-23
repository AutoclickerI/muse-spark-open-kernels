"""Optional Python call observer. Records metadata; never substitutes model operations."""
import json
import os
import sys
import time

_NAMES = {'_row_split_gemm', '_reduce_scatter_rows_gemm', '_forward_experts',
          '_forward_marlin', 'forward'}
_FILES = {'muse_spark.py', 'muse_spark_bf16_moe.py', 'muse_spark_fp4_moe.py'}
_SEEN = set()


def profile(frame, event, result):
    if event != 'return' or frame.f_code.co_name not in _NAMES:
        return
    code = frame.f_code
    if os.path.basename(code.co_filename) not in _FILES:
        return
    import torch

    loc = frame.f_locals
    tensor = next((loc[k] for k in ('x', 'partial', 'hidden_states')
                   if k in loc and isinstance(loc[k], torch.Tensor)), None)
    obj = loc.get('self')
    layer = getattr(obj, 'layer_idx', None)
    shape = tuple(tensor.shape) if tensor is not None else None
    capture = torch.cuda.is_current_stream_capturing()
    key = (code.co_filename, code.co_name, layer, shape, capture)
    if key in _SEEN:
        return
    _SEEN.add(key)
    rec = dict(timestamp=time.time(), file=code.co_filename, function=code.co_name, layer=layer,
               input_shape=shape, capturing=capture,
               returned_shape=list(result.shape) if isinstance(result, torch.Tensor) else None)
    if code.co_name == '_forward_experts':
        rec.update(router_dtype=str(obj.gate.weight.dtype), top_k=obj.top_k,
                   expert_type=type(obj.experts).__name__,
                   use_marlin=getattr(obj.experts, 'use_marlin', None))
    directory = os.environ['MUSE_HARNESS_TRACE']
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f'{os.getpid()}.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')


sys.setprofile(profile)
