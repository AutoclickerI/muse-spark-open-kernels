set -e
cd "$(dirname "$0")"
for f in ${@:-cu256 cu128}; do
  for a in ${ARCHS:-90a 120}; do
    if nvcc -std=c++20 -O3 -lineinfo -gencode=arch=compute_$a,code=sm_$a -cubin $f.cu -o ${f}_$a.cubin -Xptxas -v > ptxas_${f}_$a.log 2>&1; then echo "$f $a ok"; else echo "$f $a FAIL"; grep -m20 -i "error" ptxas_${f}_$a.log; fi
  done
done
