"""ABBA host-overhead experiment in one server; all switches are probe-only."""
import argparse
import hashlib
import json
import os
import re
import signal
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving/harness"))
import common
import run
import run_trace  # instrument only the server child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="host-ab2")
    parser.add_argument("--tree", default="/data2/jihoon/muse_kernel/harness/trees/host-final")
    parser.add_argument("--step-ab", action="store_true")
    options = parser.parse_args()
    root = common.RESULTS / options.prefix
    root.mkdir()
    control = root / "mode"
    control.write_text("baseline")
    tree = Path(options.tree)
    target, draft = common.model_dirs("margin")
    repo = Path(__file__).resolve().parents[1]
    env = [f"PYTHONPATH={repo / 'prof'}:{tree / 'python'}:{repo / 'serving/harness/pyhook'}",
           f"MUSE_SERVING_TRACE={root / 'host_trace.jsonl'}", "MUSE_SERVING_NSYS=1",
           "MUSE_TRACE_GRAPH_EVENTS=1", "MUSE_TRACE_REQUESTS=120",
           f"MUSE_HOST_AB={control}",
           "MUSE_HOST_AB_BEFORE=/data2/jihoon/muse_kernel/harness/trees/host-before/python/sglang/srt/managers/overlap_utils.py"]
    if options.step_ab:
        env = [item for item in env if not item.startswith(("MUSE_HOST_AB", "MUSE_TRACE_REQUESTS", "MUSE_TRACE_GRAPH_EVENTS"))]
        env += ["MUSE_HOST_STEP_AB=8", "MUSE_TRACE_REQUESTS=30"]
    args = argparse.Namespace(target=str(target), draft=str(draft), tp=1, gpus="0",
                              mem_fraction_static=0.3, max_total_tokens=300000,
                              attention_backend=None, extra=[], env=env)
    port = run.free_port()
    proc, cmd = run.launch(args, tree, root, port)
    workload = run.build_workload(0, [1, 4], {1: 2, 4: 8}, 3, [2600])
    metas = []
    try:
        startup = run.wait_ready(proc, port, root / "server.log", 1200)
        modes = ["patch"] if options.step_ab else ["baseline", "patch", "patch", "baseline"]
        for phase, mode in enumerate(modes):
            # The server waits for its outstanding overshoot step to drain.
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/flush_cache?timeout=30", timeout=60) as response:
                assert response.status == 200
            control.write_text(mode)
            label = f"{options.prefix}-{phase}-{mode}"
            out = common.RESULTS / label
            out.mkdir()
            (out / "workload.jsonl").write_text("".join(json.dumps(w)+"\n" for w in workload))
            print(label, flush=True)
            records = []
            for level in [1, 4]:
                records += run.run_level(port, level, [w for w in workload if w["level"] == level], 600)
            (out / "requests.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            meta = dict(label=label, weights="margin", tp=1, gpus="0", server_cmd=cmd,
                        provenance=run.tree_provenance(tree), startup_s=startup,
                        checkpoint_sha256={"target":common.checkpoint_fingerprint(target), "draft":common.checkpoint_fingerprint(draft)},
                        workload=dict(seed=0, levels=[1,4], convs={1:2,4:8}, turns=3, first_turn_lens="2600", requests=30,
                                      sha256=hashlib.sha256(json.dumps([w["input_ids"] for w in workload]).encode()).hexdigest()),
                        sampling_params=dict(temperature=0.0,top_k=1,max_new_tokens=220,ignore_eos=True),
                        summary=run.summarize(records,[1,4]), validation_errors=[])
            (out / "meta.json").write_text(json.dumps(meta,indent=2))
            metas.append((out,meta))
            if not all(v["all_220"] and not v["errors"] for v in meta["summary"].values()):
                raise RuntimeError("Incomplete AB workload")
    finally:
        # Preserve the Nsight collector while terminating only its application.
        # Killing the collector's process group can lose the final report.
        text = (root / "server.log").read_text()
        match = re.search(r"Started server process \[(\d+)\]", text)
        if match:
            try:
                os.kill(int(match[1]), signal.SIGTERM)
                proc.wait(timeout=60)
            except (ProcessLookupError, run.subprocess.TimeoutExpired):
                run.stop(proc)
        else:
            run.stop(proc)
        proc.harness_logger.join()
        log = run.scan_log(root / "server.log")
        for out,meta in metas:
            meta["log"] = log
            meta["validation_errors"] = run.graph_errors(log,1,[])
            (out / "meta.json").write_text(json.dumps(meta,indent=2))


if __name__ == "__main__":
    main()
