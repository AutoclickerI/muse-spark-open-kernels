# Attention P5: retain current decode selections

No new decode attention cell is promoted.

- Full q8: retain FA3. be19da94 producer427.8/713.7us at B16/B32 loses;
  combine6.3/8.5us is small. sm90a SASS proves that its FP8 mma.sync is
  emulated with conversions plus FP16 HMMA;223 registers allow only two
  CTAs/SM. No spills. See `README.md` and `sass_evidence.json`.
- Verify1 sliding:32f2f07a beats the board's attn_main_twog profile12.1vs16.2us,
  but best7 uses public FA3. The guarded adapter remains disabled until its
  paired FA3 result lands. Source: `muse_decode_layer/dev/verify1_sliding_pick.py`,
  SHA46b940320048047cace11d16807e160c11cdd84f364ff902cd04bacf81ee263e.
- Combined proj lab send `e0712c33ab1cb715` is held until09:00:20 KST on
  2026-10-25 at priority0.8. Payload sections `verify8_producer_phases` and
  `verify1_sliding_fa3_pair`. No judge ID before dispatch. The first measures
  producer phases and an explicit stage-widening variant; the second pairs
  the adapter with public FA3 plus current/best5 layer glue, including
  FP32 projection-partial ABIs. Native H200 runtime and whole-layer schedule
  effects remain unverified.
- Attention candidate9828bfa041a143a9 still needs its receipt before enabling
  conditional-rescale or deferred-denominator switches in a new candidate.
