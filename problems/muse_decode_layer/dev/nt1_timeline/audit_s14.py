"""Run local_eval normal/forced with S9 dispatch checks and a complete retained-tensor inventory."""
import importlib.util
import json
import sys
import types
from pathlib import Path

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
sys.path.insert(0, str(R / 'tools'))
import local_eval


def main():
    import static_cache
    static_cache.install(local_eval)
    forced = '--forced' in sys.argv
    output_arg = Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else H
    candidate = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else H.parents[1] / 'cand_r3_best7_s14.py'
    out = output_arg / ('forced' if forced else 'normal')
    out.mkdir(exist_ok=True, parents=True)
    spec = importlib.util.spec_from_file_location('s9_audit', H.parent / 's9/audit.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    original_inventory = audit.inventory
    original_loader = local_eval.load_module
    module = []
    counts = {}

    def loader(name, path):
        m = original_loader(name, path)
        if name == 'submission':
            module.append(m)
            if hasattr(m, 'cd_gemv'):
                gemv = m.cd_gemv
                def counted(x, c, S, **kwargs):
                    key = f"T{x.shape[0]}:N{c['N']}:K{c['K']}:S{S}"
                    row = counts.setdefault(key, dict(eager=0, captured=0))
                    row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager'] += 1
                    return gemv(x, c, S, **kwargs)
                m.cd_gemv = counted
            def wrap(fn, prefix):
                def counted(x, c, S, **kwargs):
                    key=f"{prefix}:T{x.shape[0]}:N{c['N']}:K{c['K']}:S{S}"
                    row=counts.setdefault(key,dict(eager=0,captured=0))
                    row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
                    return fn(x,c,S,**kwargs)
                return counted
            if hasattr(m,'dcp_gemv'):m.dcp_gemv=wrap(m.dcp_gemv,'direct')
            if hasattr(m,'gi_run'):
                original_gi=m.gi_run
                def counted_gi(x,w,s,cfg):
                    key=f'bal2:T{x.shape[0]}';row=counts.setdefault(key,dict(eager=0,captured=0))
                    row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
                    return original_gi(x,w,s,cfg)
                m.gi_run=counted_gi
        return m

    def inventory(torch, state, weights):
        for name,c in state.get('direct_coded',{}).items():
            c={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in c.items()}
            state['direct_coded'][name]=c;state[name]=state[name].clone()
            expected=dict(coded=torch.uint8,origin=torch.int32,eoff=torch.int32,esc=torch.int32,base=torch.int32)
            assert {k:v.dtype for k,v in c.items() if isinstance(v,torch.Tensor)}==expected
            assert not bool((c['base']&1).any())
            assert torch.equal(c['origin'],(c['base']>>1)*0x01010101)
            assert torch.equal(module[0].dcp_decode(c).view(torch.int16),state[name].view(torch.int16)),name
        rows, per = original_inventory(torch, state, weights)
        tensors, seen, objects = {}, {}, set()
        def walk(path, value):
            if isinstance(value, torch.Tensor):
                key = (value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape), value.dtype)
                row = dict(dtype=str(value.dtype).removeprefix('torch.'), shape=list(value.shape),
                           bytes=value.numel() * value.element_size())
                if key in seen:row['alias_of'] = seen[key]
                else:seen[key] = path
                if value.device.type == 'cpu':
                    row['holds'] = 'CPU launch metadata; no retained GPU weight values'
                elif '.direct_coded.' in path:
                    field=path.rsplit('.',1)[-1]
                    row['holds']=dict(coded='lossless 12-bit BF16 codes',origin='int32 repeated exponent-origin byte per 16 rows; no widened values',base='int32 even exponent-window base per 16 rows',esc='int32 coordinates and original BF16 16-bit escape patterns',eoff='int32 escape offsets')[field]
                elif 'coded_bank' in path or '.coded_dense.' in path:
                    field = path.rsplit('.', 1)[-1]
                    row['holds'] = dict(coded='lossless 12-bit BF16 sign/mantissa/exponent-window codes',
                        scales='FP32 exact power of two exponent-window metadata; no per-weight widened values',
                        esc='int32 coordinate and original 16-bit BF16 bit pattern for each escape',
                        eoff='int32 escape offsets per tile', base='int32 exponent-window base per 16 rows').get(field, 'metadata')
                elif any(s in path for s in ('w13', 'w2')):
                    last = path.rsplit('.', 1)[-1]
                    if last.endswith(('_gs', 'gs13', 'gs2', 'weight_scale_2')) or last in ('gs13', 'gs2'):
                        row['holds'] = 'original FP32 global expert scale'
                    elif '_sf' in last or 'scale' in last:
                        row['holds'] = 'original E4M3 group-scale codes or lossless byte recode/permutation'
                        assert value.dtype in (torch.uint8, torch.float8_e4m3fn), (path, row)
                    elif value.dtype in (torch.uint8, torch.int32) or str(value.dtype).startswith('torch.float4'):
                        row['holds'] = 'packed E2M1 expert nibbles, optionally interleaved with lossless E4M3 recodes'
                    else:
                        raise AssertionError(('unclassified expert weight tensor', path, row))
                elif path.rsplit('.', 1)[-1] in ('blob13', 'blob2'):
                    row['holds'] = 'raw17 lossless packed E2M1 code permutation'
                    assert value.dtype == torch.uint8, (path, row)
                elif path.rsplit('.', 1)[-1] in ('sfb13', 'sfb2'):
                    row['holds'] = 'raw17 original E4M3 group-scale byte permutation'
                    assert value.dtype == torch.uint8, (path, row)
                elif value.dtype == torch.bfloat16:
                    row['holds'] = 'original BF16 dense weight/norm, BF16 router hi/lo, or synthetic BF16 scratch'
                else:
                    row['holds'] = 'original router/norm/global-scale precision, indices, or synthetic scratch; inspect path'
                tensors[path] = row
                return
            if id(value) in objects:return
            objects.add(id(value))
            if isinstance(value, dict):
                for k, v in value.items():walk(f'{path}.{k}', v)
            elif isinstance(value, (tuple, list)):
                for i, v in enumerate(value):walk(f'{path}[{i}]', v)
            elif isinstance(value, types.FunctionType) and value.__closure__:
                for i, c in enumerate(value.__closure__):walk(f'{path}.closure[{i}]', c.cell_contents)
            elif hasattr(value, '__dict__') and not isinstance(value, (types.ModuleType, type)):
                for k, v in vars(value).items():walk(f'{path}.{k}', v)
        walk('state', state)
        for name, c in state.get('coded_dense', {}).items():
            assert torch.equal(module[0].cd_decode(c).view(torch.int16), state[name].view(torch.int16)), name
        if hasattr(module[0], '_CD_BANK'):
            walk('coded_bank', module[0]._CD_BANK)
            for c in module[0]._CD_BANK.values():
                # The patch already asserts this in prepare; audit independently against its source role.
                role = next(n for n in ('qkv_gate', 'pool_router') if tuple(state[n].shape) == (c['N'], c['K']))
                assert torch.equal(module[0].cd_decode(c).view(torch.int16), state[role].view(torch.int16)), role
        (out / 'precision_inventory.json').write_text(json.dumps(tensors, indent=2) + '\n')
        return rows, per

    local_eval.load_module = loader
    audit.inventory = inventory
    sys.argv = ['audit.py', '--candidate', str(candidate), '--out', str(out)] + (['--forced'] if forced else [])
    try:audit.main()
    finally:
        (out / 'coded_calls.json').write_text(json.dumps(counts, indent=2) + '\n')
        print('S14_CALLS', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
