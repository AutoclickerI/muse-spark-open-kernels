# okc: problem=muse_bf16_proj
import torch
def prepare(weights):raise RuntimeError('lab-only host: evidence completed at import')
def kernel_fn(*args):raise RuntimeError('lab-only host')
kernel=kernel_fn

_LAB_CONFIG = {'host': 'muse_bf16_proj', 'phase': 'import', 'import_budget': 30.0, 'base_sha256': '84984bc733d61a770d60e638b8e71da2ee56ce56735d280a2846c6114aad9b7f', 'lab_only': True, 'base_prepare_s': 130, 'output_budget': 12000}
_LAB_RESOURCES = [{'compile_s': 0, 'run_s': 3, 'output_bytes': 8744, 'context_poison': 'none'}, {'compile_s': 0, 'run_s': 3, 'output_bytes': 1024, 'context_poison': 'none'}, {'compile_s': 0, 'run_s': 3, 'output_bytes': 512, 'context_poison': 'none'}]
_lab_packet_ns = {}
exec('"""Wire format for lab evidence. This stdlib-only module is embedded in judge wrappers."""\nimport base64\nimport hashlib\nimport json\nimport zlib\n\n\ndef encode_packet(name, source_sha, packet):\n    data = json.dumps(packet, sort_keys=True, separators=(\',\', \':\'), ensure_ascii=True, allow_nan=False).encode()\n    checksum = hashlib.sha256(data).hexdigest()\n    blob = base64.b85encode(zlib.compress(data, 9)).decode()\n    return f\'LAB_PACKET {name} {source_sha} {checksum} {blob}\'\n\n\ndef read_packets(tail):\n    packets = []\n    for line in tail.splitlines():\n        if not line.startswith(\'LAB_PACKET \'):\n            continue\n        try:\n            _, name, source_sha, checksum, blob = line.split(\' \', 4)\n            data = zlib.decompress(base64.b85decode(blob))\n            if hashlib.sha256(data).hexdigest() != checksum:\n                raise ValueError(\'checksum mismatch\')\n            packets.append(dict(name=name, source_sha256=source_sha, checksum=checksum, packet=json.loads(data)))\n        except Exception as exc:\n            raise ValueError(\'invalid lab result packet: \'+str(exc)) from exc\n    return packets\n\n\ndef result_lines(tail):\n    """Recover legacy result lines losslessly, or the structured evidence JSON."""\n    packets = read_packets(tail)\n    if not packets:\n        return tail.splitlines()  # archived pre-packet receipts\n    lines = []\n    for record in packets:\n        packet = record[\'packet\']\n        lines.extend(packet[\'lines\'] if \'lines\' in packet else [json.dumps(packet, sort_keys=True)])\n    return lines\n\n\ndef paired_evidence(tail, kind):\n    """Read structured integration evidence or an archived XZ85 result line.\n\n    A wrapper\'s post-measurement synchronization can discover a sticky device fault.\n    Preserve that failure when handing inner paired data to the promotion evaluator.\n    """\n    import lzma\n    def with_transport(raw, packet):\n        errors = packet.get(\'transport_errors\', [])\n        if errors:\n            raw[\'complete\'] = False\n            raw.setdefault(\'failures\', []).append(dict(scope=\'device\', reason=\' | \'.join(errors)))\n        return raw\n    prefix = kind+\'_XZ85 \'\n    records = read_packets(tail)\n    for record in records:\n        packet = record[\'packet\']\n        if packet.get(\'kind\') == kind:\n            return with_transport(packet[\'paired\'], packet)\n        for line in packet.get(\'lines\', []):\n            if line.startswith(prefix):\n                raw = json.loads(lzma.decompress(base64.b85decode(line[len(prefix):])))\n                return with_transport(raw, packet)\n    lines = [line[len(prefix):] for line in tail.splitlines() if line.startswith(prefix)]\n    if not lines:\n        return None\n    if len(lines) != 1:\n        raise ValueError(\'ambiguous paired evidence for \'+kind)\n    return json.loads(lzma.decompress(base64.b85decode(lines[0])))\n', _lab_packet_ns)
_lab_encode_packet = _lab_packet_ns['encode_packet']

# ---- h200_lab wrapper: payloads p1_maximum 79aea13813e66a1a, p1_optional 384c9385e1afe31b, p1_peer 7d4b3459a37155f4 on base lab-only muse_bf16_proj ----
_LAB_PAYLOADS = [('p1_maximum', '79aea13813e66a1a', "LAB_RESOURCES={'compile_s': 0, 'run_s': 3, 'output_bytes': 8744, 'context_poison': 'none'}\nimport torch,time\n\ndef native_pairs():\n    seeds=[901,1901]\n    pairs=[]\n    # Native CUDA work proves the payloads execute on the receipt's actual device.\n    # This is a transport probe, never a kernel promotion or correctness claim.\n    for seed in seeds:\n        gen=torch.Generator(device='cuda');gen.manual_seed(seed)\n        x=torch.randn(1024,device='cuda',generator=gen)\n        us=[]\n        for _ in range(2):\n            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)\n            start.record();torch.add(x,x);end.record();end.synchronize()\n            us.append(round(start.elapsed_time(end)*1000,4))\n        pairs.append(dict(seed=seed,old_us=us[0],new_us=us[1]))\n    return dict(identities=dict(candidate='torch.add transport probe',incumbent='torch.add transport probe'),\n                seeds=seeds,paired=pairs,checks=dict(gpu=torch.cuda.get_device_name(),\n                cc=list(torch.cuda.get_device_capability()),TASK='not applicable: transport probe'),complete=True)\n\nimport base64,hashlib,json,random,string,zlib\n\ndef lab(deadline,weights,state):\n    packet=native_pairs()\n    packet['padding']=''\n    packet['padding_sha256']=''\n    rng=random.Random(20261026)\n    data=''.join(rng.choices(string.ascii_letters+string.digits,k=20000))\n    def fit(n):\n        packet['padding']=data[:n]\n        packet['padding_sha256']=hashlib.sha256(packet['padding'].encode()).hexdigest()\n        raw=json.dumps(packet,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()\n        blob=base64.b85encode(zlib.compress(raw,9)).decode()\n        return len('LAB_PACKET p1_maximum '+'0'*16+' '+hashlib.sha256(raw).hexdigest()+' '+blob+'\\n')\n    lo,hi=0,len(data)\n    while lo<hi:\n        mid=(lo+hi+1)//2\n        if fit(mid)<=8744:lo=mid\n        else:hi=mid-1\n    for n in range(max(0,lo-128),min(len(data),lo+128)+1):\n        if fit(n)==8744:break\n    else:raise RuntimeError('could not fill exact maximum packet allocation')\n    print('P1_NOISY_IMPORT '+'x'*18000,flush=True)\n    return dict(packet=packet,logs=['P1 maximum packet 8744 bytes','diagnostics '+'x'*18000])\n"), ('p1_optional', '384c9385e1afe31b', "LAB_RESOURCES={'compile_s': 0, 'run_s': 3, 'output_bytes': 1024, 'context_poison': 'none'}\ndef lab(deadline,weights,state):\n    raise RuntimeError('P1 expected isolated optional-arm failure; no device fault')\n"), ('p1_peer', '7d4b3459a37155f4', "LAB_RESOURCES={'compile_s': 0, 'run_s': 3, 'output_bytes': 512, 'context_poison': 'none'}\nimport torch,time\n\ndef native_pairs():\n    seeds=[901,1901]\n    pairs=[]\n    # Native CUDA work proves the payloads execute on the receipt's actual device.\n    # This is a transport probe, never a kernel promotion or correctness claim.\n    for seed in seeds:\n        gen=torch.Generator(device='cuda');gen.manual_seed(seed)\n        x=torch.randn(1024,device='cuda',generator=gen)\n        us=[]\n        for _ in range(2):\n            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)\n            start.record();torch.add(x,x);end.record();end.synchronize()\n            us.append(round(start.elapsed_time(end)*1000,4))\n        pairs.append(dict(seed=seed,old_us=us[0],new_us=us[1]))\n    return dict(identities=dict(candidate='torch.add transport probe',incumbent='torch.add transport probe'),\n                seeds=seeds,paired=pairs,checks=dict(gpu=torch.cuda.get_device_name(),\n                cc=list(torch.cuda.get_device_capability()),TASK='not applicable: transport probe'),complete=True)\n\ndef lab(deadline,weights,state):\n    return dict(packet=native_pairs(),logs=['P1 peer ran after the failed optional payload'])\n")]
_lab_ran = False
_lab_flush_buf = None
_lab_output = []


def _lab_print(line):
    _lab_output.append(line)


def lab_cold_l2():
    # The judge's cold-L2 step (TASK.md): read a buffer larger than L2 before each timed replay. Never flush with
    # zero_()/fill_(): that leaves ~50 MB of dirty lines for the timed call to write back (+5-16 us at M=1-32).
    global _lab_flush_buf
    if _lab_flush_buf is None:
        _lab_flush_buf = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    _lab_flush_buf.view(torch.int64).sum()  # int64 view: the uint8 sum upcasts and runs ~4x slower


def _lab_run_one(name, src, deadline, weights, state):
    import inspect as _inspect, linecache as _linecache
    fname = f"<h200_lab {name}>"  # in linecache, so inspect.getsource (triton.jit) finds the source
    _linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = dict(globals())
    exec(compile(src, fname, "exec"), ns)
    fn = ns["lab"]
    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)


def _lab_run(weights, state, _start, _end, _phase):
    global _lab_ran
    import time as _t
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        _records = []
        _reserve = sum(r["output_bytes"] + 256 for r in _LAB_RESOURCES)
        _log_cap = max(0, 12000 - _reserve) // len(_LAB_PAYLOADS)
        for _i, (_name, _sha, _src) in enumerate(_LAB_PAYLOADS):
            _t0 = _t.monotonic()
            _res = _LAB_RESOURCES[_i]
            _need = _res["compile_s"] + _res["run_s"]
            _remaining = sum(r["compile_s"]+r["run_s"] for r in _LAB_RESOURCES[_i:])
            _deadline = min(_end, _t0 + max(_need, max(0., _end-_t0)*_need/_remaining))
            _lines = []
            _packet = None
            try:
                if _t0 >= _end:
                    raise TimeoutError("lab budget exhausted before this payload")
                _result = _lab_run_one(_name, _src, _deadline, weights, state)
                if isinstance(_result, dict):
                    _packet = _result["packet"]
                    if not all(k in _packet for k in ("identities", "seeds", "paired", "checks", "complete")):
                        raise ValueError("packet needs identities, seeds, paired, checks, complete")
                    _lines = [str(x) for x in _result.get("logs", [])]
                else:
                    _lines = [str(x) for x in (_result or [])]
                    _packet = dict(lines=_lines, complete=None, format="legacy-lines")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except BaseException as _e:
                _errors = [f"LAB_ERROR {type(_e).__name__}: {_e}"] + _tb.format_exc().splitlines()[-8:]
                _lines += _errors
                if _packet is None:
                    _packet = dict(lines=_lines, complete=False)
                else:
                    _packet = dict(_packet, complete=False, transport_errors=_errors)
            try:
                _wire = _lab_encode_packet(_name, _sha, _packet)
            except Exception as _e:
                _wire = _lab_encode_packet(_name, _sha, dict(complete=False, lines=[
                    f"LAB_PACKET_ERROR {type(_e).__name__}: {str(_e)[:100]} "]))
            if len(_wire.encode()) + 1 > _res["output_bytes"]:
                # Never cut a packet. The planner needs a larger declaration or a smaller experiment.
                _wire = _lab_encode_packet(_name, _sha, dict(complete=False, lines=[
                    f"LAB_PACKET_OVERSIZE needs {len(_wire.encode())+1} bytes; declared {_res['output_bytes']}; split payload"]))
            _diagnostics = []
            _used = 0
            for _line in _lines:
                if _used + len(_line.encode()) + 1 <= _log_cap:
                    _diagnostics.append(_line)
                    _used += len(_line.encode()) + 1
            _records.append([f"LAB_BEGIN {_name} {_sha}"] + _diagnostics + [_wire,
                f"LAB_END lab_s={_t.monotonic()-_t0:.1f} {_phase}_s={_t.monotonic()-_start:.1f}"])
        # Emit all reserved packets after every payload (and, on import hosts, after base prepare).
        # A later compiler's stdout cannot evict an earlier packet from the judge's receipt tail.
        for _record in _records:
            for _line in _record:
                _lab_print(_line)

import time as _lab_time
_lab_import_start = _lab_time.monotonic()
_lab_run(None, None, _lab_import_start, _lab_import_start + 30.0, "import")

_lab_base_prepare = prepare


def prepare(weights):
    try:
        return _lab_base_prepare(weights)
    finally:
        # Emit after the host tuner, whose output can exceed the judge's entire log tail.
        for _line in _lab_output:
            print(_line, flush=True)
        _lab_output.clear()
