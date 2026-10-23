"""Collect the harness teacher-forced probe before an existing server stops."""

import json


def collect(run, args, tree, out, port):
    from teacher_forced import request, token_logprobs

    k = 8
    prompt = run.build_workload(0, [1], {1: 1}, 1, [2600])[0]["input_ids"]
    base = request(
        port,
        dict(
            input_ids=prompt,
            sampling_params=dict(
                temperature=0.0, top_k=1, max_new_tokens=k, ignore_eos=True
            ),
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=0,
        ),
    )
    continuation = list(base.get("output_ids") or [])[:k]
    if len(continuation) != k:
        raise RuntimeError("teacher-forced probe did not return eight tokens")
    forced_prompt = prompt + continuation
    forced = request(
        port,
        dict(
            input_ids=forced_prompt,
            sampling_params=dict(
                temperature=0.0, top_k=1, max_new_tokens=1, ignore_eos=True
            ),
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=0,
        ),
    )
    lps = token_logprobs(forced.get("meta_info", {}))
    ids = [
        row[1] for row in forced.get("meta_info", {}).get("input_token_logprobs", [])
    ]
    if len(lps) != len(forced_prompt) or lps[0] is not None or ids != forced_prompt:
        raise RuntimeError("teacher-forced probe token/logprob alignment failed")
    record = dict(
        label=out.name,
        tree=str(tree),
        weights=args.weights,
        tp=args.tp,
        prompt_len=len(prompt),
        k=k,
        prompt_ids=prompt,
        generated_ids=continuation,
        generated_output_logprobs=base.get("meta_info", {}).get(
            "output_token_logprobs", []
        ),
        prompt_input_logprobs=lps,
        forced_generated_token_logprobs=lps[len(prompt) :],
        forced_generated_token_ids=ids[len(prompt) :],
    )
    (out / "teacher_forced.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        "Teacher-forced eight-token probe saved:",
        out / "teacher_forced.json",
        flush=True,
    )
