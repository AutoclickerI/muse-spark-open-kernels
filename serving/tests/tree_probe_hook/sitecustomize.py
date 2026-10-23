"""Read-only DFlash lattice recorder; never changes proposals or verification."""

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
exec(
    compile(
        (ROOT / "serving/harness/pyhook/sitecustomize.py").read_text(),
        "harness_hook",
        "exec",
    )
)
TARGET = "sglang.srt.speculative.dflash_worker_v2"


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        import hashlib
        import struct
        import torch

        lattice = module._selector_lattice
        choose = module.DFlashWorkerV2._choose_suffix_candidates
        records = Path(os.environ["MUSE_TREE_PROBE_DIR"])
        records.mkdir(parents=True, exist_ok=True)
        counter = 0
        prompts = {}
        prompt_lengths = {}

        def record_lattice(model, hidden, anchor):
            ids, scores = lattice(model, hidden, anchor)
            if not hasattr(model, "_tree_probe_lattices"):
                model._tree_probe_lattices = {}
            model._tree_probe_lattices[hidden.shape[0]] = (ids, scores, anchor)
            return ids, scores

        def record_choose(worker, batch, draft_next):
            nonlocal counter
            result = choose(worker, batch, draft_next)
            bs = len(batch.reqs)
            bank = getattr(worker.draft_model, "_tree_probe_lattices", {})
            choices = sorted(n for n in bank if n >= bs)
            if not choices:
                raise RuntimeError("tree probe saw no selector lattice for live batch")
            # Verify the retained graph-buffer ABI against the actual proposal.
            # This catches stale/padded bucket selection instead of recording it.
            target = draft_next.detach().cpu()
            selected = None
            for size in choices:
                ids, scores, anchor = (t[:bs].detach().cpu() for t in bank[size])
                idx = scores[:, 0, 0].argmax(-1)
                path = []
                for depth in range(ids.shape[1]):
                    if depth:
                        idx = scores[torch.arange(bs), depth, idx].argmax(-1)
                    path.append(ids[torch.arange(bs), depth, idx])
                if torch.equal(torch.stack(path, 1), target):
                    selected = ids, scores, anchor
                    break
            if selected is None:
                raise RuntimeError(
                    "tree probe retained lattice does not reproduce live chain"
                )
            ids, scores, anchor = selected
            positions = (
                worker._draft_block_positions_buf[:bs, 0].detach().cpu().tolist()
            )
            rows = []
            for i, req in enumerate(batch.reqs):
                if req.rid not in prompts:
                    prompt = list(req.origin_input_ids)
                    digest = hashlib.sha256(
                        struct.pack(f"<{len(prompt)}I", *prompt)
                    ).hexdigest()
                    prompts[req.rid] = digest
                    prompt_lengths[req.rid] = len(prompt)
                rows.append(
                    dict(
                        rid=req.rid,
                        prompt_hash=prompts[req.rid],
                        output_prefix=list(req.output_ids),
                        anchor=int(anchor[i]),
                        position=int(positions[i]),
                        prompt_len=prompt_lengths[req.rid],
                    )
                )
            torch.save(
                dict(rows=rows, ids=ids, scores=scores, chain=target),
                records / f"{os.getpid()}-{counter:06d}.pt",
            )
            counter += 1
            return result

        module._selector_lattice = record_lattice
        module.DFlashWorkerV2._choose_suffix_candidates = record_choose
        print("TREE_PROBE read-only lattice recording enabled", flush=True)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == TARGET:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(spec.loader)
            return spec


sys.meta_path.insert(0, Finder())
