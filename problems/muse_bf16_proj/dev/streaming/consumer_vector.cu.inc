    // Two MMA steps consume one 16-byte shared load per operand row. The
    // builder uses BK+32 shared rows: each eight-lane transaction visits the
    // first and second rows in opposite halves of the bank array.
    // The K permutation (eight adjacent BF16s per lane) is identical for W
    // and X, so this changes summation order, not values or precision.
#pragma unroll
    for (int k = 0; k < BK; k += 32) {
#pragma unroll
      for (int c = 0; c < CELLS; ++c) {
        const int cell = warp + c * WARPS;
        const int rn = (cell / (T / 8)) * 16, tm = (cell % (T / 8)) * 8;
        const B* wa = s + (rn + g) * STRIDE + k + l * 8;
        const B* xb = s + (R + tm + g) * STRIDE + k + l * 8;
        uint4 a0 = *reinterpret_cast<const uint4*>(wa);
        uint4 a1 = *reinterpret_cast<const uint4*>(wa + 8 * STRIDE);
        uint4 b = *reinterpret_cast<const uint4*>(xb);
        mma(a[c], a0.x, a1.x, a0.y, a1.y, b.x, b.y);
        mma(a[c], a0.z, a1.z, a0.w, a1.w, b.z, b.w);
      }
    }
