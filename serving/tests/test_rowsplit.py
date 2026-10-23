"""2-GPU NCCL test of muse_spark._row_split_gemm and _reduce_scatter_rows_gemm against the replicated math.

Run: tools/img env PYTHONPATH=$PWD/serving/sglang/python python -m torch.distributed.run --nproc_per_node 2 serving/tests/test_rowsplit.py
Result 2026-10-23 on 2x RTX PRO 6000: max |diff| 0 at m = 1024, 1027, 2049, 16384."""
import os, torch, torch.distributed as dist
import sglang.srt.models.muse_spark as ms

class Shim:  # the GroupCoordinator surface the helpers use
    def __init__(self): self.world_size, self.rank_in_group = dist.get_world_size(), dist.get_rank()
    def all_gather_into_tensor(self, out, inp): dist.all_gather_into_tensor(out, inp)
    def reduce_scatter_tensor(self, out, inp): dist.reduce_scatter_tensor(out, inp)

def main():
    rank = int(os.environ["RANK"]); torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    ms.get_tp_group = lambda: Shim()
    ms.bf16_gemm_dispatch = lambda x, w, b: torch.nn.functional.linear(x, w, b)
    g = torch.Generator(device="cuda").manual_seed(0)
    for m in (1024, 1027, 2049, 16384):
        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16, generator=g)
        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16, generator=g) / 20
        ref = torch.nn.functional.linear(x, w)
        got = ms._row_split_gemm(x, w)
        torch.manual_seed(rank + 1)
        part = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        full = part.clone(); dist.all_reduce(full)
        ref2 = torch.nn.functional.linear(full, w)
        got2 = ms._reduce_scatter_rows_gemm(part, w)
        e1 = (got.float() - ref.float()).abs().max().item(); e2 = (got2.float() - ref2.float()).abs().max().item()
        if rank == 0:
            print(f"m={m}: pooler-style shape {tuple(got.shape)} max|diff| {e1:.3g}; unpooler-style shape {tuple(got2.shape)} max|diff| {e2:.3g} (ref scale {ref2.abs().max().item():.3g})")
    dist.destroy_process_group()

main()
