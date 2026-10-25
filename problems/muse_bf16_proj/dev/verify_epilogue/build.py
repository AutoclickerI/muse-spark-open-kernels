"""Build self-contained P7 payloads from frozen S11 final and the lane kernels."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
S11 = REPO / 'problems/muse_decode_layer/cand_r3_best7_s11_final.py'
S11_SHA = 'acda6c62'
CONFIGS = [(bm, bn, bk, st, nw, 1) for bm, bn, bk, st, nw in (
    (32, 128, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 64, 4, 4),
    (64, 128, 64, 6, 4), (64, 128, 128, 3, 4), (64, 256, 64, 4, 8),
    (64, 256, 128, 2, 8), (128, 64, 64, 4, 4), (128, 128, 64, 4, 8))]


def pieces(src, names):
    lines = src.splitlines()
    out = []
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            first = min([d.lineno for d in n.decorator_list] + [n.lineno])
            out.append('\n'.join(lines[first-1:n.end_lineno]))
    assert len(out) == len(names)
    return '\n\n'.join(out)


def write(name, text):
    compile(text, name, 'exec')
    path = HERE / name
    path.write_text(text)
    print(path.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())


def main():
    src = S11.read_text()
    assert hashlib.sha256(src.encode()).hexdigest().startswith(S11_SHA)
    kernels = (HERE / 'kernels.py').read_text()
    common = kernels + '\n\n' + pieces(src, ('_pdl_start', '_pdl_wait', '_boundary_fast', '_boundary_late'))
    write('lab_verify_sweep.py', common + '\n\n' +
          (HERE / 'sweep_body.py').read_text().replace('__CONFIGS__', repr(CONFIGS)))
    timing = pieces((HERE / 'sweep_body.py').read_text(), ('ve_capture', 've_timing'))
    write('lab_verify_stats.py', common + '\nimport statistics\n' + timing + '\n\n' + (HERE / 'stats_body.py').read_text())
    if (HERE / 'layer_body.py').exists():
        from patch_verify_epilogue import run_block
        host = run_block(src) + '\n\n' + kernels
        # Reuse the paired-lab helpers, not its old experiments.
        helper_src = (REPO / 'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc').read_text()
        helpers = pieces(helper_src, ('ab_label', 'ab_namespace', 'ab_copy', 'ab_graph', 'ab_tensors', 'ab_gate', 'ab_time', 'ab_summary'))
        imports = 'import gc\nimport math\nimport statistics\nimport time\nimport types\nimport linecache\nimport torch\n'
        prefix = imports + '\nVE_HOST_SRC = ' + repr(host) + '\n\n' + helpers + '\n\n'
        layer = (HERE / 'layer_body.py').read_text()
        write('lab_verify_layer.py', prefix + layer)
        # The first H200 receipt rejects combined GEMM/fusion. Attribute the
        # faster M128 six-stage plain GEMM to each role against actual Lt.
        ls=layer.splitlines()
        labnode=next(n for n in ast.parse(layer).body if isinstance(n,ast.FunctionDef) and n.name=='lab')
        rownode=next(n for n in ast.walk(labnode) if isinstance(n,ast.Assign) and
                     isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rows')
        replacement="""        cfg6=VE_CFG6 if torch.cuda.get_device_capability()==(9,0) else VE_CFG
        rows={'A':{},'A2':{},
              'Go4':{(m,'o'):(VE_CFG,8192,False)},
              'Go6':{(m,'o'):(cfg6,8192,False)},
              'Gu4':{(m,'unpooler'):(VE_CFG,8192,False)},
              'Gu6':{(m,'unpooler'):(cfg6,8192,False)}}"""
        ls[rownode.lineno-1:rownode.end_lineno]=replacement.splitlines()
        gemm='\n'.join(ls)+'\n'
        gemm=gemm.replace("        if torch.cuda.get_device_capability() != (9, 0):\n            rows.pop('F6')\n",'')
        first=gemm.index("    lines = [");last=gemm.index('\n    host, st =',first)
        gemm=gemm[:first]+"    lines = ['META S11 final acda6c62 plain-GEMM attribution Go/Gu=O/unpooler; 4/6=stages; A2=A:A; 12 seeds x6 cold READ; r=variant/A']"+gemm[last:]
        write('lab_verify_gemm.py',prefix+gemm)


if __name__ == '__main__':
    main()
