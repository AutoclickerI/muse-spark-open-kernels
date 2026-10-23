"""Compare the chain with an exact maximum-mass prefix tree of eight nodes.

One node is the verified anchor, leaving seven proposal nodes. Prefix reach
probabilities decrease down every edge, so best-first selection of the seven
largest probabilities is globally optimal among prefix-closed trees. Scores
are drafter probabilities, not counterfactual target logits. Actual coverage is
checked against the same request's final greedy continuation.
"""

import argparse
import hashlib
import heapq
import json
import math
import struct
from pathlib import Path


def distribution(values, temperature):
    maximum = max(values)
    exp = [math.exp((v - maximum) / temperature) for v in values]
    total = sum(exp)
    return [v / total for v in exp]


def optimal_tree(probabilities, budget=7):
    heap = [(-p, (i,)) for i, p in enumerate(probabilities[0][0])]
    heapq.heapify(heap)
    nodes = []
    mass = 0.0
    while heap and len(nodes) < budget:
        negative, path = heapq.heappop(heap)
        mass -= negative
        nodes.append(path)
        depth = len(path)
        if depth < len(probabilities):
            for i, p in enumerate(probabilities[depth][path[-1]]):
                heapq.heappush(heap, (negative * p, path + (i,)))
    return nodes, 1 + mass


def chain_path(scores):
    path, previous = [], 0
    for depth in range(len(scores)):
        previous = max(
            range(len(scores[depth][previous])), key=scores[depth][previous].__getitem__
        )
        path.append(previous)
    return tuple(path)


def chain_expectation(probabilities, path):
    total, reach, previous = 1.0, 1.0, 0
    for depth, index in enumerate(path):
        reach *= probabilities[depth][previous][index]
        total += reach
        previous = index
    return total


def coverage(nodes, ids, future):
    best = 0
    for path in nodes:
        if len(path) <= len(future) and all(
            ids[d][i] == future[d] for d, i in enumerate(path)
        ):
            best = max(best, len(path))
    return min(1 + best, len(future))


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    args = ap.parse_args()
    requests = {
        r["key"]: r
        for r in map(
            json.loads, (args.result / "requests.jsonl").read_text().splitlines()
        )
    }
    workload = list(
        map(json.loads, (args.result / "workload.jsonl").read_text().splitlines())
    )
    by_hash = {}
    for item in workload:
        prompt = item["input_ids"]
        key = hashlib.sha256(struct.pack(f"<{len(prompt)}I", *prompt)).hexdigest()
        by_hash[key] = requests[item["key"]]
    temperatures = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0)
    aggregate = {}
    position = {
        d: dict(
            n=0,
            q1=0.0,
            q2=0.0,
            entropy=0.0,
            actual_hits=0,
            conditional_n=0,
            conditional_hits=0,
        )
        for d in range(7)
    }
    records = []
    excluded_prompts = {}
    offsets = {}
    request_blocks = {}
    terminal_blocks = 0
    for file in sorted((args.result / "lattices").glob("*.pt")):
        batch = torch.load(file, map_location="cpu", weights_only=False)
        for i, meta in enumerate(batch["rows"]):
            if meta["prompt_hash"] not in by_hash:
                excluded_prompts[meta["prompt_hash"]] = (
                    excluded_prompts.get(meta["prompt_hash"], 0) + 1
                )
                continue
            request = by_hash[meta["prompt_hash"]]
            prefix = meta["output_prefix"]
            output = request["output_ids"]
            assert output[: len(prefix)] == prefix, (
                file,
                request["key"],
                "prefix drift",
            )
            # Overlap scheduling leaves req.output_ids one result behind.
            # Recover the anchor position recursively from the previous actual
            # accepted path, and check every next anchor against the final stream.
            request_blocks[request["key"]] = request_blocks.get(request["key"], 0) + 1
            offset = offsets.get(meta["rid"], 0)
            if offset is None:
                terminal_blocks += 1
                continue
            if "position" in meta:
                assert offset == meta["position"] - meta["prompt_len"], (
                    file,
                    "position drift",
                )
            assert output[offset] == meta["anchor"], (
                file,
                request["key"],
                offset,
                "anchor offset",
            )
            future = output[offset + 1 :]
            ids = batch["ids"][i].tolist()
            scores = batch["scores"][i].float().tolist()
            path = chain_path(scores)
            chain = [ids[d][idx] for d, idx in enumerate(path)]
            assert chain == batch["chain"][i].tolist()
            chain_nodes = [path[:d] for d in range(1, 8)]
            actual_chain = coverage(chain_nodes, ids, future)
            offsets[meta["rid"]] = offset + actual_chain
            if len(future) < 8:
                offsets[meta["rid"]] = None
                terminal_blocks += 1
                continue  # right-censored terminal block, reported separately
            row = dict(
                key=request["key"],
                level=request["level"],
                offset=offset,
                chain=actual_chain,
            )
            for temperature in temperatures:
                probabilities = [
                    [distribution(v, temperature) for v in edges] for edges in scores
                ]
                tree, expected_tree = optimal_tree(probabilities)
                expected_chain = chain_expectation(probabilities, path)
                actual_tree = coverage(tree, ids, future)
                group = aggregate.setdefault(
                    (request["level"], temperature),
                    dict(
                        n=0,
                        chain=0,
                        tree=0,
                        expected_chain=0.0,
                        expected_tree=0.0,
                        changed=0,
                    ),
                )
                group["n"] += 1
                group["chain"] += actual_chain
                group["tree"] += actual_tree
                group["expected_chain"] += expected_chain
                group["expected_tree"] += expected_tree
                group["changed"] += set(tree) != set(chain_nodes)
                row[str(temperature)] = dict(
                    tree=actual_tree,
                    expected_chain=expected_chain,
                    expected_tree=expected_tree,
                    nodes=tree,
                )
                if temperature == 1.0:
                    previous, alive = 0, True
                    for d, idx in enumerate(path):
                        probs = probabilities[d][previous]
                        hit = ids[d][idx] == future[d]
                        stats = position[d]
                        stats["n"] += 1
                        stats["q1"] += probs[idx]
                        stats["q2"] += sum(sorted(probs, reverse=True)[:2])
                        stats["entropy"] -= sum(
                            p * math.log(max(p, 1e-300)) for p in probs
                        )
                        stats["actual_hits"] += hit
                        stats["conditional_n"] += alive
                        stats["conditional_hits"] += alive and hit
                        alive = alive and hit
                        previous = idx
            records.append(row)
    assert set(request_blocks) == set(requests), "missing request lattice trace"
    for key, count in request_blocks.items():
        assert count >= requests[key]["spec_verify_ct"], (key, "missing verify blocks")
    report = dict(
        requests=len(requests),
        all_220=all(r.get("n_out") == 220 for r in requests.values()),
        uncensored_blocks=len(records),
        fixed_nodes=8,
        proposal_nodes=7,
        positions={},
        results=[],
        excluded_prompt_blocks=excluded_prompts,
        request_blocks=request_blocks,
        right_censored_blocks=terminal_blocks,
    )
    for d, values in position.items():
        n = values["n"]
        report["positions"][str(d + 1)] = dict(
            top1_probability=values["q1"] / n,
            top2_probability=values["q2"] / n,
            entropy_nats=values["entropy"] / n,
            actual_top1_hit=values["actual_hits"] / n,
            actual_conditional_hit=values["conditional_hits"]
            / max(1, values["conditional_n"]),
            conditional_n=values["conditional_n"],
        )
    for (level, temperature), values in sorted(aggregate.items()):
        n = values["n"]
        report["results"].append(
            dict(
                level=level,
                temperature=temperature,
                blocks=n,
                chain_tau=values["chain"] / n,
                tree_tau=values["tree"] / n,
                actual_gain=values["tree"] / values["chain"] - 1,
                model_chain_tau=values["expected_chain"] / n,
                model_tree_tau=values["expected_tree"] / n,
                model_gain=values["expected_tree"] / values["expected_chain"] - 1,
                changed_tree_fraction=values["changed"] / n,
            )
        )
    (args.result / "tree_ceiling.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.result / "tree_blocks.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
