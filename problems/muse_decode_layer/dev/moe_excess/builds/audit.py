"""Native instruction/resource witnesses for the actual shipped extension."""
import json
import re
import subprocess
import candidates as build

H = build.H
sass = (H / 'out/night.sass').read_text()
chunks = re.split(r'\n\s*Function : ', sass)[1:]
names = [part.splitlines()[0].strip() for part in chunks]
demangled = subprocess.check_output(['c++filt'], input='\n'.join(names) + '\n', text=True).splitlines()
rows = []
for name, part in zip(demangled, chunks):
    if not re.search(r'pb_\w+::kernel<', name):
        continue
    counts = {k: len(re.findall(pattern, part)) for k, pattern in dict(
        HGMMA=r'\bHGMMA\.', LDGSTS=r'\bLDGSTS\.', LDG=r'\bLDG\.',
        BF16_product=r'\b(?:HMUL2|HFMA2)[^;\n]*BF16', LDS=r'\bLDS\.',
        STS=r'\bSTS\.', DEPBAR=r'\bDEPBAR\.', WARPGROUP=r'\bWARPGROUP\.'
    ).items()}
    assert counts['HGMMA'] > 0 and counts['BF16_product'] > 0, (name, counts)
    rows.append(dict(name=name, counts=counts))
assert len(rows) == 22, len(rows)
compile_rows = []
for part in re.split("ptxas info    : Compiling entry function '", (H / 'out/native.log').read_text())[1:]:
    name = subprocess.check_output(['c++filt', part.split("'", 1)[0]], text=True).strip()
    if not re.search(r'pb_\w+::kernel<', name):
        continue
    spill = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', part)
    registers = re.search(r'Used (\d+) registers', part)
    assert spill and registers
    compile_rows.append(dict(name=name, registers=int(registers[1]), stack_bytes=int(spill[1]),
                             spill_store_bytes=int(spill[2]), spill_load_bytes=int(spill[3])))
assert len(compile_rows) == 22
manifest = json.loads((H / 'candidates.json').read_text())
assert manifest['native_cuda_sha256'] == build.sha((H / 'out/night.cu').read_bytes())
report = dict(passed=True, arch='sm90a', cuda_sha256=manifest['native_cuda_sha256'],
              sass_sha256=build.sha((H / 'out/night.sass').read_bytes()), kernels=rows,
              resources=compile_rows, source_witnesses={
                  'REG': 'Source directly loads original Q/SC packed words into registers, only X is cp.async staged. One WG, 64-row tiles, no widened prepare values.',
                  'DECODE': 'One 16-register fragment replaces two; wait_group0 completes the asynchronous read before reuse. Exact dq unchanged.',
                  'PAD': 'm64n48k16 WGMMA emitted; exact remaining rows choose16/32/48/64. Same weight bytes, precision and K reduction.',
                  'OCC': '128threads,WG1,3stages,32-row tiles,38400 dynamic shared bytes. Residency advantage is a resource bound pending H200.',
                  'BALANCE': 'Shared prefix over C/ORDER, uniform expert,row64,channel task index. No host route read or output caching.',
                  'EPI': 'Original scale bits/global scales, early norm loads; finalizer14 is existing warp-only PDL implementation, no CTA shared exchange.'},
              limitations='Instruction counts are static. No H200 occupancy or performance is inferred; EPI W2 retains incumbent spill4/4 at depth4.')
(H / 'native_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print('native', len(rows), 'kernels audited')
