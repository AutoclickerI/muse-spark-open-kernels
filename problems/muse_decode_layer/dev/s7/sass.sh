set -e
cd "$(dirname "$0")"
for f in hh_s6 hh_s7; do
  nvcc -std=c++20 -O3 -lineinfo -gencode=arch=compute_90a,code=sm_90a -cubin $f.cu -o $f.cubin -Xptxas -v > $f.ptxas.log 2>&1
  cuobjdump -sass $f.cubin > $f.sass
done
