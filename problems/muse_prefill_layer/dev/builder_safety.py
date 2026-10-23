"""Shared builder transform: guarded tuner lifecycle, independent numerical admission.

Queued candidate files are immutable. Builders may emit a new file through --output;
never run one against an already-queued output unless reconstructing it deliberately.
"""
import ast
from pathlib import Path

def guarded_tuners(source):
    safe=Path(__file__).with_name('safe_score_tuning.py').read_text()
    names={n.name for n in ast.parse(safe).body if isinstance(n,ast.FunctionDef)}
    lines=source.splitlines()
    ranges=[(n.lineno,n.end_lineno) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    for a,b in sorted(ranges,reverse=True):del lines[a-1:b]
    result='\n'.join(lines)+'\n'+safe
    if 'def _fused_flush_l2(' not in source:
        flush_source=Path(__file__).with_name('fused_expert_paths.py').read_text()
        for node in ast.parse(flush_source).body:
            if isinstance(node,ast.FunctionDef) and node.name in ('_fused_flush_read','_fused_flush_l2'):
                start=min([node.lineno]+[d.lineno for d in node.decorator_list])
                result+='\n'+'\n'.join(flush_source.splitlines()[start-1:node.end_lineno])+'\n'
    result=result.replace("json.dumps(st.pop('_measurements'),separators=(',',':'))", "json.dumps(_score_compact_table(st.pop('_measurements')),separators=(',',':'))")
    ast.parse(result)
    return result
