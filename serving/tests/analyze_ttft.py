"""Join request-keyed server CUDA/host timelines to the read-only harness output."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DATA = Path("/data2/jihoon/muse_kernel/harness/results")


def analyze(label, level=16):
    root = DATA / label
    events = [
        json.loads(x) for x in (root / "ttft_server.jsonl").read_text().splitlines()
    ]
    by_rid = defaultdict(list)
    for event in events:
        if event["kind"] != "prefill":
            continue
        for req in event["reqs"]:
            by_rid[req["rid"]].append((event, req))
    rows = []
    for request in map(json.loads, (root / "requests.jsonl").read_text().splitlines()):
        if request["level"] != level:
            continue
        chunks = by_rid[request["trace_rid"]]
        assert chunks, request["key"]
        first, last = chunks[0], chunks[-1]
        admitted = first[1]["admitted"]
        finished = last[0]["finished"][request["trace_rid"]]
        assert finished > admitted > request["trace_start"]
        # Prefill GPU event intervals include the target, draft-context KV
        # materialization and stream waits. Scheduling/chunk gaps are queue.
        gpu_end = last[0]["gpu_end"]
        gpu = sum(e["gpu_ms"] for e, r in chunks) / 1000
        prefill = gpu
        emit = request["trace_first"] - gpu_end
        queue = gpu_end - request["trace_start"] - prefill
        draft = (
            sum(
                s["gpu_ms"]
                for e, r in chunks
                for s in e["stages"]
                if s["name"] == "_append_target_hidden_to_draft_kv_by_loc"
            )
            / 1000
        )
        row = {
            k: request[k]
            for k in ("key", "level", "conv", "turn", "ttft", "cached_tokens")
        }
        row.update(
            queue_s=queue,
            prefill_execution_s=prefill,
            first_emit_s=emit,
            admission_wait_s=admitted - request["trace_start"],
            prefill_finish_cpu_s=finished - gpu_end,
            prefill_gpu_s=gpu,
            draft_kv_gpu_s=draft,
            prefill_cpu_launch_s=sum(e["launched"] - e["before"] for e, r in chunks),
            result_wait_s=sum(e["process_enter"] - e["launched"] for e, r in chunks),
            chunks=len(chunks),
            shapes=[e["tokens"] for e, r in chunks],
        )
        assert abs(queue + prefill + emit - request["ttft"]) < 1e-5
        rows.append(row)
    summary = {}
    for turn in (None, 0, 1, 2):
        rs = [r for r in rows if turn is None or r["turn"] == turn]
        summary[str(turn)] = {
            k: statistics.mean(r[k] for r in rs)
            for k in (
                "ttft",
                "queue_s",
                "prefill_execution_s",
                "first_emit_s",
                "prefill_gpu_s",
                "draft_kv_gpu_s",
                "prefill_cpu_launch_s",
                "result_wait_s",
                "admission_wait_s",
                "prefill_finish_cpu_s",
            )
        }
    return {"label": label, "summary_means": summary, "per_request": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--level", type=int, default=16)
    args = parser.parse_args()
    data = [analyze(label, args.level) for label in args.labels]
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    for item in data:
        print(item["label"], json.dumps(item["summary_means"], indent=2))
