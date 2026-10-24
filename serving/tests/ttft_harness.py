"""Run the read-only integration harness, optionally recording TTFT timelines."""

import http.client
import json
import sys
import threading
import time

import integration_harness as integration
import run

trace = "--trace-ttft" in sys.argv
if trace:
    sys.argv.remove("--trace-ttft")
    original_launch = integration.wrapper.launch.__globals__["launch"]

    def launch(args, tree, out, port):
        args.mem_fraction_static = 0.85 if args.tp == 1 else 0.3
        args.max_total_tokens = 300000
        if args.tp > 1:
            args.env += ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0"]
        portable = not integration.wrapper.native and tree != run.common.STOCK_TREE
        args.env += [
            f"PYTHONPATH={tree / 'python'}:{integration.ROOT / 'serving/tests/ttft_hook'}",
            f"MUSE_TTFT_TRACE={out / 'ttft_server.jsonl'}",
        ]
        if portable:
            args.env += ["MUSE_TTFT_PORTABLE=1"]
        return original_launch(args, tree, out, port)

    run.launch = launch
    original_request = http.client.HTTPConnection.request
    local = threading.local()

    def request(self, method, url, body=None, headers=None, **kwargs):
        if url == "/generate" and body:
            local.start = time.perf_counter()
            import hashlib

            data = json.loads(body)
            data["rid"] = (
                "ttft-"
                + hashlib.sha256(json.dumps(data["input_ids"]).encode()).hexdigest()[
                    :24
                ]
            )
            local.rid = data["rid"]
            body = json.dumps(data)
        return original_request(self, method, url, body, headers or {}, **kwargs)

    http.client.HTTPConnection.request = request
    original_stream = run.stream_generate

    def stream(port, ids, timeout):
        rec = original_stream(port, ids, timeout)
        # request() is entered immediately after the harness's t0; using
        # this point excludes JSON construction that its TTFT also excludes.
        rec.update(
            trace_rid=local.rid,
            trace_start=local.start,
            trace_first=local.start + rec["ttft"],
        )
        return rec

    run.stream_generate = stream

if __name__ == "__main__":
    run.main()
