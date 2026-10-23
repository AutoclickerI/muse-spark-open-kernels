import sys, torch, json
sys.path.insert(0, "problems/muse_prefill_layer")
import reference as ref
w = ref.generate_weights(ref.WEIGHTS_SEED)
spec = json.load(open("upstream/api/problems/muse_prefill_layer.json"))
for b in spec["benchmarks"]:
    case = dict(b)
    if case["q_len"] * case["batch"] > 4000:
        case["q_len"] = 4096  # memory: estimate with a smaller block
    data = ref.generate_input(**{**case, "local_start": 0})
    front = ref.ref_front(data, w)
    ids = front["biased"].topk(8, dim=-1).indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ls = 32 * int(pairs.argmax())
    local = (ids - ls) % 256
    loc = local[local < 32]
    cnt = torch.bincount(loc, minlength=32)
    T = ids.shape[0]
    print(b["batch"], b["ctx"], b["layer"], b["q_len"], "T", T, "local routes", int(cnt.sum()), f"({cnt.sum().item()/T:.2f}/tok)", "touched", int((cnt>0).sum()), "max", int(cnt.max()), "sorted", cnt.sort(descending=True).values[:8].tolist())
    del data, front
    torch.cuda.empty_cache()
