"""Compile the direct V9 reader for Hopper without loading its CUDA entry."""
import hashlib
from torch.utils.cpp_extension import load_inline
from sglang.kernels.ops.moe.muse_marlin_v9_reader import reader_source
from sglang.srt.layers.muse_spark_moe import CUDA_SRC,CPP_SRC

source=reader_source(CUDA_SRC)
tag=hashlib.sha256(source.encode()).hexdigest()[:12]
load_inline(name=f'muse_marlin_v9_reader_sm90_{tag}',cpp_sources=CPP_SRC,cuda_sources=source,
            functions=['max_smem_attr','smem_needed','resident_blocks','run'],
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo'],verbose=False)
print('DIRECT READER sm90a COMPILE PASS',tag,flush=True)
