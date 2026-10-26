# operatorstack/open-kernel-challenge-solutions @ 6d9b77656b6c295e421be9e1424b0f8f99299b0a
# MIT License
#
# Copyright (c) 2026 bigboateng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Open Kernel Challenge solutions (team bigboateng)
# Copyright (c) 2026 bigboateng. Licensed under the MIT License (see LICENSE),
# except for the portions listed below.
#
# This repository uses and reproduces code from SGLang
# (https://github.com/sgl-project/sglang), Copyright the SGLang authors,
# licensed under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0). The SGLang-derived portions
# listed below remain under the Apache License 2.0. The solutions also import
# SGLang (sglang, sgl_kernel) at run time; those packages are not included here.
#
# muse_moe_w4a16/submission.py
#   - Imports _post_expert_norm_weighted_sum_kernel from
#     sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py.
#   - _wsum_ks reproduces that kernel's body (same math, order and rsqrt), with
#     one change: it first sums FP32 split-K partials and rounds them to BF16.
#
# muse_bf16_proj/submission.py
#   - Imports _bf16_gemm_dispatch_impl from
#     sglang/srt/layers/quantization/unquant.py (the reference path for every
#     shape this file does not serve with its own kernels).
#   - The router's hi/lo split with FP32 accumulation follows
#     MuseSpark1xRouter.forward_split2.
#
# muse_decode_layer/submission.py
#   - _post_expert_ws_pdl_kernel reproduces the body of SGLang's
#     _post_expert_norm_weighted_sum_kernel
#     (sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py),
#     with programmatic dependent launch added, and with an optional sum of FP32
#     split-K partials before the norm.
#   - _router_hilo uses the same hi/lo split with FP32 accumulation as
#     MuseSpark1xRouter.forward_split2.
#   - _rms_norm_local_ids_kernel reimplements SGLang's single-CTA
#     moe_align_small_numel algorithm (same "+1 offset" buckets) inside the
#     pre-expert norm kernel.
#   - Imports SGLang modules and kernels: _post_expert_norm_weighted_sum_kernel,
#     silu_and_mul, moe_wna16_marlin_gemm, FusedMuseSparkRouting,
#     FusedRMSNormGate, FusedPostNormResidualPreNorm, fused_norm_residual,
#     moe_align_block_size, MuseSpark1xNormPreservingResidualConnection,
#     MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale,
#     marlin_make_workspace, prepare_moe_nvfp4_layer_for_marlin,
#     _bf16_gemm_dispatch_impl, and sgl_kernel's flash_attn_with_kvcache and
#     scalar_types.
#
# muse_attention_decode/submission.py
#   - Imports sgl_kernel's flash_attn_with_kvcache and SGLang's rms_norm_gate
#     (sglang/kernels/ops/muse_spark_v12/norm_gate.py).
#
# lee101/openkernels @ 601c65413496cf63f942b4f15ca087839ef2f5cc; MoE v43d
# Modified by Jihoon: reviewed kernel extraction, explicit ABI/phase launchers, no donor tuning or diagnostics.
# No upstream Lee NOTICE file exists at this pin.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright [yyyy] [name of copyright owner]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch
_ROT = ((0, 0), (3, 1), (6, 6), (-5, 5))
def _place(word, code, pair, hi):
    de, ds = _ROT[pair]
    for bit, tgt, d in ((0, 6, de), (1, 7, de), (2, 8, de), (3, 15, ds)):
        word |= ((code >> bit) & 1) << ((tgt + 16 * hi - d) % 32)
    return word

def _repack(w, sf, perm):
    """w uint8 [R, K/2], sf e4m3 [R, K/16] -> uint8 [R/16, K/256, 2304]: per 16-row tile and 256-K half stage, 4 x 512 B of
    mma A fragments (lane-major uint4) then 32 lanes x 8 B of block scales."""
    R, K = w.shape[0], w.shape[1] * 2
    MT, NS = R // 16, K // 512
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(MT, 16, NS, 8, 4, 4, 4)
    a, b = C[:, :8], C[:, 8:]
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=w.device)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 3, 1, 4, 5).contiguous().view(torch.uint8).view(MT, NS, 2, 4, 512)
    S = s8.to(torch.uint8).view(MT, 16, NS, 8, 4)
    sc = torch.stack((S[:, :8], S[:, 8:]), -1).permute(0, 2, 1, 4, 3, 5).contiguous().view(MT, NS, 32, 2, 8)
    sc = sc.permute(0, 1, 3, 2, 4).contiguous()
    return torch.cat((word.view(MT, NS, 2, 2048), sc.view(MT, NS, 2, 256)), -1).contiguous().view(MT, NS * 2, 2304)

def _repack2(w, sf):
    """W2 for the fused path: uint8 [64 rowblocks, 128 granules (32 K), 4 rowtiles, 288]: per 16-row tile and 32-K granule,
    32 lanes x 2 words (mma A fragments of two k16 steps; lane t holds K 8t..8t+7) then 32 B of block scales [g][blk][row g / g+8].
    Rows permuted inside each tile (mma row i < 8 -> 2i, i >= 8 -> 2(i-8)+1) so a lane's C fragment holds adjacent output rows."""
    R, K = w.shape[0], w.shape[1] * 2
    RT, NG = R // 16, K // 32
    dev = w.device
    pi = torch.tensor([2 * i for i in range(8)] + [2 * i + 1 for i in range(8)], device=dev)
    perm = (torch.arange(RT, device=dev)[:, None] * 16 + pi[None, :]).reshape(-1)
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(RT, 16, NG, 4, 2, 4)
    a, b = C[:, :8], C[:, 8:]
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=dev)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 1, 3, 4).contiguous().view(torch.uint8).view(RT, NG, 256)
    S = s8.to(torch.uint8).view(RT, 2, 8, NG, 2).permute(0, 3, 2, 4, 1).contiguous().view(RT, NG, 32)
    rec = torch.cat((word, S), -1).view(RT // 4, 4, NG, 288).permute(0, 2, 1, 3).contiguous()
    return rec

def _repackX(w):
    """[E, tiles, 16 half-stages, 2304] -> [E, tiles / 8, 64 chunks, 8 tiles, 576]: per 16-row tile and 64-K chunk the
    512 B of mma A-fragment words (lane-major uint4) then the 32 lanes x 2 B block scales (row g, row g + 8)"""
    E, T = w.shape[0], w.shape[1]
    words = w[..., :2048].reshape(E, T, 64, 512)
    sc = w[..., 2048:].reshape(E, T, 16, 32, 4, 2).permute(0, 1, 2, 4, 3, 5).reshape(E, T, 64, 64)
    return torch.cat((words, sc), -1).view(E, T // 8, 8, 64, 576).permute(0, 1, 3, 2, 4).contiguous()
# Own host wrapper; the donor autotuner/diagnostics/driver are not imported.
def prepare_lee(weights):
 dev=weights['w13'].device;E=32;I=4096
 perm1=torch.stack((torch.arange(I,device=dev).view(-1,8),I+torch.arange(I,device=dev).view(-1,8)),1).reshape(-1)
 perm2=torch.arange(I,device=dev)
 w1=torch.stack([_repack(weights['w13'][e],weights['w13_sf'][e],perm1) for e in range(E)])
 w2=torch.stack([_repack(weights['w2'][e],weights['w2_sf'][e],perm2) for e in range(E)])
 w2f=torch.stack([_repack2(weights['w2'][e],weights['w2_sf'][e]) for e in range(E)])
 return dict(w1=w1,w2f=w2f,w1x=_repackX(w1),w2x=_repackX(w2),g1=weights['w13_gs'][:E].float()*256.,g2=weights['w2_gs'][:E].float()*256.,nw=1.+weights['post_norm_w'].float())

def lee_plan(data,state,mod,kind='X',split=False):
 x=data['hidden_states'];M=x.shape[0];dev=x.device;ids=data['topk_ids'];tw=data['topk_weights'];lo=data['config']['local_start'];out=torch.empty_like(x)
 if kind=='F':
  yacc=torch.zeros((M*8,4096),device=dev);ctr=torch.zeros(9280+64,dtype=torch.int32,device=dev);prof=torch.empty(0,dtype=torch.int64,device=dev)
  ts=[x,ids,tw,state['w1'],state['w2f'],state['g1'],state['g2'],state['nw'],yacc,ctr,out,prof];ptr=[t.data_ptr() for t in ts]
  def fused():mod.port_f(ptr,M,lo,torch.cuda.current_stream().cuda_stream)
  steps=[('fused',fused)]
 else:
  rows=M*8+32*7;xg=torch.empty(rows*8192,dtype=torch.uint8,device=dev);ig=torch.empty_like(xg);y=torch.empty((M*8,4096),dtype=torch.bfloat16,device=dev);ss=torch.empty(M*8,device=dev);ctr=torch.zeros(64,dtype=torch.int32,device=dev);prof=torch.empty(0,dtype=torch.int64,device=dev)
  ts=[x,ids,tw,state['w1x'],state['w2x'],state['g1'],state['g2'],state['nw'],xg,ig,y,ss,ctr,out,prof];ptr=[t.data_ptr() for t in ts]
  def call(mode):mod.port_x(ptr,M,lo,mode,torch.cuda.current_stream().cuda_stream)
  steps=[('route',lambda:call(0))]+([('w13',lambda:call(2)),('w2',lambda:call(3)),('finish',lambda:call(4))] if split else [('fused',lambda:call(1))])
 return dict(steps=steps,out=out,owners=ts)
