"""Reproduce the standalone fixed cell from the source-pinned S4 definitions."""
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]
sys.path.insert(0,str(HERE.parents[1]))
from build_prefill_lab import subset


def main():
    s=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s4.py').read_text()
    text=subset(s,('issue_prepare','issue_word_bank','issue_call'))
    text+='\n\ndef d32g_prepare(weights):\n    return issue_word_bank(issue_prepare(dict(weights,post_norm_w=weights.get("post_expert_norm",weights.get("post_norm_w")))))\n\ndef d32g_experts(x,scores,local_ids,bank,local_start=0):\n    assert x.shape[0]==128\n    return issue_call(dict(hidden_states=x,topk_ids=local_ids,topk_weights=scores,config=dict(local_start=local_start)),bank,"idist",1,1056)\n'
    sha=hashlib.sha256(text.encode()).hexdigest()
    assert sha=='a7afb63d9cb17c920412c8f3003640bf956599ab6caa874a9631f290798dce82',sha
    (HERE/'d32g_cell.py').write_text(text);print(sha)


if __name__=='__main__':main()
