#!/bin/bash
set -euo pipefail
xl_dir=problems/muse_decode_layer/dev/moe_excess
for xl_name in ladder ladder_pm; do
 tools/img /usr/local/cuda/bin/nvcc -std=c++20 --expt-relaxed-constexpr -O3 -lineinfo -Xptxas=-v -gencode arch=compute_90a,code=sm_90a -c "$xl_dir/out/$xl_name.cu" -o "$xl_dir/out/$xl_name.o" > "$xl_dir/out/$xl_name.compile.log" 2>&1
 /usr/local/cuda/bin/cuobjdump --dump-sass "$xl_dir/out/$xl_name.o" > "$xl_dir/out/$xl_name.sass"
done
