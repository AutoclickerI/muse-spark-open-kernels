"""Alternate pristine-stock and patch requests with both servers resident on GPU0."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving/harness"))
import common
import run


def main():
    root = common.RESULTS / "host-dual"
    root.mkdir()
    target, draft = common.model_dirs("margin")
    servers = {}
    metas = []
    workload = run.build_workload(0, [1, 4], {1: 2, 4: 8}, 3, [2600])
    try:
        for mode, tree in [("stock", common.STOCK_TREE), ("patch", Path("/data2/jihoon/muse_kernel/harness/trees/host-final"))]:
            out = root / mode
            out.mkdir()
            args = argparse.Namespace(target=str(target), draft=str(draft), tp=1, gpus="0",
                                      mem_fraction_static=0.6, max_total_tokens=100000,
                                      attention_backend=None, extra=[], env=[])
            port = run.free_port()
            proc, cmd = run.launch(args, tree, out, port)
            servers[mode] = (proc, port, tree, out, cmd)
            run.wait_ready(proc, port, out / "server.log", 1200)
            print(mode+" ready", flush=True)
        for phase, mode in enumerate(["stock", "patch", "patch", "stock"]):
            proc, port, tree, logdir, cmd = servers[mode]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/flush_cache?timeout=30", timeout=60) as response:
                assert response.status == 200
            label = f"host-dual-{phase}-{mode}"
            out = common.RESULTS / label
            out.mkdir()
            print(label,flush=True)
            records = []
            for level in [1,4]:
                records += run.run_level(port, level, [w for w in workload if w["level"]==level], 600)
            (out/"requests.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            (out/"workload.jsonl").write_text("".join(json.dumps(w)+"\n" for w in workload))
            meta = dict(label=label,weights="margin",tp=1,gpus="0",server_cmd=cmd,
                        provenance=run.tree_provenance(tree),
                        checkpoint_sha256={"target":common.checkpoint_fingerprint(target),"draft":common.checkpoint_fingerprint(draft)},
                        sampling_params=dict(temperature=0.0,top_k=1,max_new_tokens=220,ignore_eos=True),
                        workload=dict(seed=0,levels=[1,4],convs={1:2,4:8},turns=3,first_turn_lens="2600",requests=30,
                                      sha256=hashlib.sha256(json.dumps([w["input_ids"] for w in workload]).encode()).hexdigest()),
                        summary=run.summarize(records,[1,4]),validation_errors=[])
            (out/"meta.json").write_text(json.dumps(meta,indent=2))
            metas.append((out,meta,mode))
            if not all(v["all_220"] and not v["errors"] for v in meta["summary"].values()):
                raise RuntimeError("Incomplete controlled TTFT workload")
    finally:
        for proc,port,tree,out,cmd in servers.values():
            run.stop(proc)
            proc.harness_logger.join(timeout=10)
        for out,meta,mode in metas:
            log=run.scan_log(servers[mode][3]/"server.log")
            meta["log"]=log
            meta["validation_errors"]=run.graph_errors(log,1,[])
            (out/"meta.json").write_text(json.dumps(meta,indent=2))


if __name__=="__main__":
    main()
