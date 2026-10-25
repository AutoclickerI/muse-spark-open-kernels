"""Compare the new producer with the prefill lane's raw17/masked/DG expert paths on proj."""
import ast
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def subset(source,roots):
    tree=ast.parse(source);bindings={};imports=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.Try)):imports.append(node)
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef)):bindings[node.name]=node
        elif isinstance(node,ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n,ast.Name):bindings[n.id]=node
    selected=set();pending=list(roots)
    pending.extend(n.id for imp in imports for n in ast.walk(imp) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))
    while pending:
        name=pending.pop();node=bindings.get(name)
        if node is None or id(node) in selected:continue
        selected.add(id(node))
        pending.extend(n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))
    body=[n for n in tree.body if n in imports or id(n) in selected or
          (isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for t in n.targets)
           and 'HUMMING_DISABLE_PARALLEL_BUILD' in ast.unparse(n))]
    return ast.unparse(ast.Module(body=body,type_ignores=[]))
