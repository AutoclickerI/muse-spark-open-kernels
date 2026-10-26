"""S17 revision for nominations arriving after the first scored composition."""
import ast,hashlib,importlib.util,json
from pathlib import Path
import compose
H=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('s17_prepare_constraints',H/'prepare_constraints.py')
prepare_constraints=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(prepare_constraints)
DEFAULT=['chunk4','pool','tail']


def build(enabled=None):
    enabled=set(DEFAULT if enabled is None else enabled);assert enabled<=set(DEFAULT)
    source,meta=compose.compose();parent=meta['sha256']
    raw=(H/'late_cells.json').read_bytes();m=json.loads((H/'late_cells_manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==m['sha256'];c=json.loads(raw)
    if 'chunk4' in enabled:
        cell=c['chunk4']
        # Preserve the winning P7 W2 release and immutable-metadata finish shell
        # while replacing only the heavy branch's schedule by CHUNK4.
        for old,new in [('light_grid=4\n','light_grid=4;tail=mt_module();pm=mt_pmz_module()\n'),
            ('mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)','pm.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)'),
            ('mod.br_finish(fp,m,start,4,torch.cuda.current_stream().cuda_stream)','tail.mt_finish(fp,m,start,10,torch.cuda.current_stream().cuda_stream)')]:
            assert cell.count(old)==1,(old,cell.count(old));cell=cell.replace(old,new)
        source+='\n'+cell
    if 'pool' in enabled:source+='\n'+c['pool']
    if 'tail' in enabled:
        tail=c['tail'];lines=tail.splitlines(True);edits=[]
        for node in ast.parse(tail).body:
            if isinstance(node,ast.Assign) and any(getattr(t,'id','') in ('RT_CUDA','RT_FAST_CUDA') for t in node.targets):
                cuda=ast.literal_eval(node.value)
                if '#include <vector>' not in cuda:cuda='#include <vector>\n'+cuda
                edits.append((node.lineno-1,node.end_lineno,node.targets[0].id+'='+repr(cuda)+'\n'))
        assert len(edits)==2
        for a,b,text in reversed(edits):lines[a:b]=[text]
        source+='\n'+''.join(lines)
    source+='\n_s17_late_previous_delta=_s16_prepare_delta\ndef _s16_prepare_delta(weights,state):\n    state=_s17_late_previous_delta(weights,state)\n'
    if 'chunk4' in enabled:source+='    p3b_bind(state)\n'
    if 'pool' in enabled:source+='    state=p10_pool_prepare(state)\n'
    if 'tail' in enabled:source+='    rt_module()\n'
    source+='    return state\n'
    compile(source,'<S17-late>','exec')
    original=compose.digest(source);source=prepare_constraints.apply(source)
    meta.update(parent_sha256=parent,original_candidate_sha256=original,sha256=compose.digest(source),late_enabled=sorted(enabled),late_cells_sha256=m['sha256'],prepare_routing_revision=prepare_constraints.REVISION)
    return source,meta


if __name__=='__main__':
    source,meta=build();(H/'generated/s17_late.py').write_text(source)
    (H/'late_selection.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['sha256'])
