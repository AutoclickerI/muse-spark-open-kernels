"""Wire format for lab evidence. This stdlib-only module is embedded in judge wrappers."""
import base64
import hashlib
import json
import zlib


def encode_packet(name, source_sha, packet):
    data = json.dumps(packet, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    checksum = hashlib.sha256(data).hexdigest()
    blob = base64.b85encode(zlib.compress(data, 9)).decode()
    return f'LAB_PACKET {name} {source_sha} {checksum} {blob}'


def read_packets(tail):
    packets = {}
    for line in tail.splitlines():
        if not line.startswith('LAB_PACKET '):
            continue
        try:
            _, name, source_sha, checksum, blob = line.split(' ', 4)
            data = zlib.decompress(base64.b85decode(blob))
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError('checksum mismatch')
            key = name, source_sha
            previous = packets.pop(key, None)
            if previous is not None and previous['checksum'] != checksum:
                raise ValueError('conflicting lab result packets for '+name)
            # Import evidence can print early and again after prepare. Validate every copy,
            # retaining the order of the final emission (the receipt may cut the first).
            packets[key] = dict(name=name, source_sha256=source_sha, checksum=checksum, packet=json.loads(data))
        except Exception as exc:
            raise ValueError('invalid lab result packet: '+str(exc)) from exc
    return list(packets.values())


def result_lines(tail):
    """Recover legacy result lines losslessly, or the structured evidence JSON."""
    packets = read_packets(tail)
    if not packets:
        return tail.splitlines()  # archived pre-packet receipts
    lines = []
    for record in packets:
        packet = record['packet']
        lines.extend(packet['lines'] if 'lines' in packet else [json.dumps(packet, sort_keys=True)])
    return lines


def paired_evidence(tail, kind):
    """Read structured integration evidence or an archived XZ85 result line.

    A wrapper's post-measurement synchronization can discover a sticky device fault.
    Preserve that failure when handing inner paired data to the promotion evaluator.
    """
    import lzma
    def with_transport(raw, packet):
        errors = packet.get('transport_errors', [])
        if errors:
            raw['complete'] = False
            raw.setdefault('failures', []).append(dict(scope='device', reason=' | '.join(errors)))
        return raw
    prefix = kind+'_XZ85 '
    records = read_packets(tail)
    for record in records:
        packet = record['packet']
        if packet.get('kind') == kind:
            return with_transport(packet['paired'], packet)
        for line in packet.get('lines', []):
            if line.startswith(prefix):
                raw = json.loads(lzma.decompress(base64.b85decode(line[len(prefix):])))
                return with_transport(raw, packet)
    lines = [line[len(prefix):] for line in tail.splitlines() if line.startswith(prefix)]
    if not lines:
        return None
    if len(lines) != 1:
        raise ValueError('ambiguous paired evidence for '+kind)
    return json.loads(lzma.decompress(base64.b85decode(lines[0])))
