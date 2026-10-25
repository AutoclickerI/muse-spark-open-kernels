#!/bin/sh
# Compile vq8.cu for sm_90a (inside tools/img): ptxas resources/warnings and SASS for inspection.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/build90}
mkdir -p "$out"
nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -cubin -Xptxas -v -lineinfo \
  -I/opt/sglang/lib/python3.12/site-packages/torch/include -o "$out/vq8.cubin" "$here/vq8.cu" 2>&1 | grep -v "^$" | grep -iv "declared but never\|Remark\|^ *\^\|^ *constexpr\|^ *DI " | tail -40
cuobjdump -sass "$out/vq8.cubin" > "$out/vq8.sass"
echo "SASS lines: $(wc -l < "$out/vq8.sass")"
