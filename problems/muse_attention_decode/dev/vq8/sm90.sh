#!/bin/sh
# Compile vq8.cu for sm_90a. Compilation errors or WGMMA serialization warnings fail the check.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/build90}
mkdir -p "$out"
if ! nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -cubin -Xptxas -v -lineinfo \
  -I/opt/sglang/lib/python3.12/site-packages/torch/include -o "$out/vq8.cubin" "$here/vq8.cu" > "$out/ptxas.log" 2>&1; then
  cat "$out/ptxas.log"
  exit 1
fi
cat "$out/ptxas.log"
if grep -Eq 'C75[0-9][0-9]' "$out/ptxas.log"; then
  echo 'WGMMA serialization warning: inspect the divergent path or live operand copy.' >&2
  exit 1
fi
cuobjdump -sass "$out/vq8.cubin" > "$out/vq8.sass"
echo "SASS lines: $(wc -l < "$out/vq8.sass")"
