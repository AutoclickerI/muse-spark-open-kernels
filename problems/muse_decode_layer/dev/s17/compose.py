"""Compose explicit S17 changes on the verified, frozen S16 portfolio."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1];S16=H.parent/'s16'
sys.path.insert(0,str(S16))
import patches,capture
BASE_SHA='435df5f78a0d157f979b4b472865a139a28ac333fe275b1aebc0055a76dbf6d3'
DEFAULT=['nearby','drop_dsm8','drop_pool128','large_release_only','vdsk_release','p5_verify1','p7_P']


def digest(source):return hashlib.sha256(source.encode()).hexdigest()


def frozen():
    # Another integration run may advance the mutable *_final.py while this
    # experiment runs. Its original scored snapshot is immutable.
    tree=ast.parse((R/'runs/muse_decode_layer/20261026T021523Z_submit_c6cc2874ab994ee9.py').read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    source=lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0]))).decode()
    assert digest(source)==BASE_SHA
    return source


def compose(enabled=None,*,ilp=False):
    enabled=set(DEFAULT if enabled is None else enabled)
    assert enabled<=set(DEFAULT)
    selected=json.loads((S16/'batches/504e1fa66298bf9e/selection.json').read_text())['patches']
    # Rebuild the exact frozen portfolio before changing it. Recomposition
    # deletes removed kernel bodies and their prepare work as well as dispatch.
    base,meta=patches.compose(selected);assert meta['sha256']==BASE_SHA and base==frozen()
    removed=set()
    if 'drop_dsm8' in enabled:removed.add('dsm_8')
    if 'drop_pool128' in enabled:removed.add('layer_library_pool128')
    if 'large_release_only' in enabled:removed.add('layer_release_moe')
    if 'p5_verify1' in enabled:removed.add('dsm_16')
    specs=[s for s in selected if s['name'] not in removed]
    if 'vdsk_release' in enabled:specs.append(patches.registry()['layer_release_projections'])
    if 'nearby' in enabled:specs.append(patches.registry()['layer_nearby'])
    source,_=patches.compose(specs)
    manifest=json.loads((H/'cells_manifest.json').read_text())
    payload=(H/'cells.json').read_bytes();assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
    cells=json.loads(payload);provenance=[]
    def apply(name):
        nonlocal source
        source=patches.apply_delta(source,cells[name]);provenance.append(manifest['provenance'][name])
    if 'nearby' in enabled:
        apply('guarded_nearby')
    if 'large_release_only' in enabled:
        apply('pure_large_release')
    if 'p5_verify1' in enabled:
        apply('P5_35seed_v1_DSM')
    if 'p7_P' in enabled:
        runtime=cells['p7_runtime']
        assert digest(runtime)=='0d5b0faf4c92b55691938bac09dfdad8ab3c5ccb312213b3012f073572ee7538'
        source+='\n'+runtime+'\nMT_ARM="P"\n'
        provenance.append(dict(name='P7_P',runtime_sha256=digest(runtime),receipt='1a7bbd9e33c04466',policy='best point ratio at all four verify shapes; original expert math/weight layout'))
    if ilp:
        source+='\n'+cells['ilp_cell']+'''
_s17_pair_experts=smallm_experts
def smallm_experts(data,state,flags,stages=2,grid=264):
    if data['hidden_states'].shape[0] in (8,16,32):return pilp_experts(data,state,flags,stages,grid)
    return _s17_pair_experts(data,state,flags,stages,grid)
_s16_native_builds.append(pilp_module)
_s17_before_ilp_delta=_s16_prepare_delta
def _s16_prepare_delta(weights,state):
    state=_s17_before_ilp_delta(weights,state)
    pilp_bind(state['s4_nt1_bank'],{8:69,16:69,32:69})
    return state
'''
    compile(source,'<S17>','exec')
    return source,dict(base_sha256=BASE_SHA,sha256=digest(source),enabled=sorted(enabled),removed_s16_pieces=sorted(removed),ilp_alternative=ilp,provenance=provenance)


def main():
    source,meta=compose();(H/'generated/s17.py').write_text(source)
    (H/'selection.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['sha256'],meta['enabled'])
if __name__=='__main__':main()
