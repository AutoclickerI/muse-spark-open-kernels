#pragma once
#include <stdint.h>

// Per-call, input-derived metadata. Every entry is int32, no device pointers.
// 8 sorted rows per NT1 token block; tiles are W13=512 and W2=256.
struct alignas(128) MoERouteMeta {
  int32_t version, tokens, local_start, pairs;
  int32_t count[32];
  int32_t offset[33];
  int32_t blocks8[32];
  int32_t units13[33], units2[33];
  int32_t token_local[32];
  int32_t pair_of[256]; // sorted local position -> token*8+slot, -1 unused
  int32_t pos_of[256];  // token*8+slot -> sorted local position, -1 remote
};
static_assert(sizeof(MoERouteMeta)==2944);
