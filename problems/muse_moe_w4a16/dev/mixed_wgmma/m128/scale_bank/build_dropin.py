"""Make the recoded producer alone ready for a receipt-gated best7 MoE cell."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
from build import recode_source
from build_repair_lab import pointer_bindings, rename
from native_abi import CPP_PRELUDE
from source import source


def main():
    payload = (ROOT / 'prof/lab/moe_m128_scale_bank.py').read_text()
    tree = ast.parse(payload)
    symbols = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)): symbols[node.name] = node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name): symbols[target.id] = node
    selected = set(); pending = ['scale_prepare', 'scale_call']
    while pending:
        node = symbols.get(pending.pop())
        if node is None or id(node) in selected: continue
        selected.add(id(node))
        pending.extend(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    tree.body = [n for n in tree.body if id(n) in selected]
    bindings = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    prep = bindings['scale_prepare']
    prep.args.args = prep.args.args[:1]
    for node in ast.walk(prep):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'recoded':
            node.test = ast.Constant(True)
        if isinstance(node, ast.IfExp) and isinstance(node.test, ast.Name) and node.test.id == 'recoded':
            node.test = ast.Constant(True)
    # Constant-fold preparation's lab comparison arm entirely.
    class Fold(ast.NodeTransformer):
        def visit_If(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant) and node.test.value is True: return node.body
            return node
        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant) and node.test.value is True: return node.body
            return node
    prep = Fold().visit(prep)
    native = pointer_bindings(rename(recode_source(source()), 'rot64', (3, 4, 5))).replace('rot64', 'msrec')
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [getattr(t, 'id', '') for t in node.targets]
            if 'SCALE_CUDA' in names: node.value = ast.Constant(native)
            if 'SCALE_FUNCTIONS' in names: node.value = ast.parse(repr(['msrec_' + f for f in ('init','info','route','run')]), mode='eval').body
            if 'SCALE_CPP' in names:
                node.value = ast.Constant(CPP_PRELUDE + 'int64_t msrec_init();std::vector<int64_t> msrec_info(int64_t,int64_t);void msrec_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void msrec_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);')
    bindings['scale_module'].body = [node for node in bindings['scale_module'].body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'msraw_init')]
    # Keep all symbols private to this cell when concatenated into best7.
    names = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)): names[node.name] = 'm128_scales_' + node.name.lstrip('_')
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name): names[target.id] = 'M128_SCALES_' + target.id.lstrip('_')
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = names.get(node.name, node.name); return self.generic_visit(node)
        def visit_ClassDef(self, node):
            node.name = names.get(node.name, node.name); return self.generic_visit(node)
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id); return node
    tree = ast.fix_missing_locations(Rename().visit(tree))
    text = ('"""M128 recoded fused MoE cell. Enable only after its paired H200 receipt wins."""\n'
            'import functools\nimport hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
            + ast.unparse(tree) + '\n')
    assert 'msraw' not in text and 'recoded' not in ast.unparse(prep)
    out = HERE / 'dropin.py'
    out.write_text(text)
    manifest = dict(path=str(out.relative_to(ROOT)), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        prepare='m128_scales_scale_prepare(weights)',
        call='m128_scales_scale_call(data, bank, which=3, grid=528)',
        payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        ready=False, reason='Enable only the configuration that beats paired NT1-recoded on H200 seed means and gates.',
        best7_rows=['(16, 8, 2048)', '(16, 8, 0)'])
    (HERE / 'dropin_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out.relative_to(ROOT), manifest['sha256'])


if __name__ == '__main__':
    main()
