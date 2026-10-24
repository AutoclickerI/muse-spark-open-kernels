"""Preserve stored E4M3 bytes in every inherited optional expert bank."""
import ast
import re

CHANGED = {'_pf_repack', '_pf_encode_blob_fast', '_pf_canonical_expand_coalesced', '_best6_build_adapters'}
REMOVED = {'_pf_encode_scales', '_pf_encode_sf_fast'}
SCALE = '''
// Stored E4M3 -> BF16(s * 2^118), only inside the timed device path.
__device__ __forceinline__ uint32_t muse_raw_scale118(uint32_t raw) {
  const uint32_t a=raw&127u;
  uint32_t b=a>=8 ? 0x7700u+(a<<4) : a>=4 ? 0x7680u+(a<<5) :
             a>=2 ? 0x7600u+(a<<6) : a ? 0x7600u : 0u;
  b|=(raw&128u)<<8;
  return b|(b<<16);
}
'''

def apply(source):
    original = source
    for node in reversed(ast.parse(original).body):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in REMOVED:
            text = ast.get_source_segment(original, node)
            if node.decorator_list:
                start = original.splitlines(keepends=True)
                text = ''.join(start[node.decorator_list[0].lineno-1:node.end_lineno])
            source = source.replace(text, '', 1)
        elif node.name in CHANGED:
            text = ast.get_source_segment(original, node)
            new = text
            if node.name == '_pf_repack':
                start = new.index('    u, neg, zero =')
                end = new.index('    codes = codes.reshape', start)
                new = new[:start] + '    u = sf.contiguous().view(torch.uint8)\n' + new[end:]
            elif node.name == '_pf_encode_blob_fast':
                new = re.sub(r'^        s[0-3] = .*\n', '', new, flags=re.M)
                new = re.sub(r'^(        c\d+ = .*?) \^ tl.where\(.*$', r'\1', new, flags=re.M)
                new = re.sub(r'^        c\d+ = tl.where\(.*\n', '', new, flags=re.M)
            elif node.name == '_pf_canonical_expand_coalesced':
                old = '    scale = 1879076864 + sf * 1048592'
                new = new.replace(old, '''    a = sf & 127
    b = tl.where(a >= 8, 0x7700 + (a << 4), tl.where(a >= 4, 0x7680 + (a << 5),
        tl.where(a >= 2, 0x7600 + (a << 6), tl.where(a != 0, 0x7600, 0))))
    b = b | ((sf & 128) << 8)
    scale = b | (b << 16)''')
                assert new != text
            elif node.name == '_best6_build_adapters':
                new = new.replace('b=raw[key];sf=raw[sfkey];out=torch.empty_like(b);so=torch.empty_like(sf)',
                                  'b=raw[key];sf=raw[sfkey];out=torch.empty_like(b)')
                new = re.sub(r'^            _pf_encode_sf_fast.*\n', '', new, flags=re.M)
                new = new.replace('raw[key]=out;raw[sfkey]=so', 'raw[key]=out  # scale bytes remain the raw layout permutation')
            assert new != text, node.name
            source = source.replace(text, new, 1)

    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_pf_GROUPED_SRC' for t in n.targets))
    native = ast.literal_eval(node.value)
    native = native.replace('namespace canonical {', SCALE + '\nnamespace canonical {', 1)
    assert native.count('sr[rb]=0x70007000u+(uint32_t)(sw&255)*0x100010u;') == 1
    native = native.replace('sr[rb]=0x70007000u+(uint32_t)(sw&255)*0x100010u;',
                            'sr[rb]=muse_raw_scale118(sw&255u);')
    native = native.replace('su[rb]=0x70007000u+(uint32_t)(sw>>8)*0x100010u;',
                            'su[rb]=muse_raw_scale118(sw>>8);')
    old = 'uint32_t bits = (lo ? 0x7000u + (lo << 4) : 0) | ((hi ? 0x7000u + (hi << 4) : 0) << 16);'
    assert native.count(old) == 2
    native = native.replace(old, 'uint32_t bits = (muse_raw_scale118(lo)&65535u) | (muse_raw_scale118(hi)<<16);')
    native = native.replace('// Encoded scale byte u: |sf| = 2^(u/8 - 21) * (1 + (u%8)/8).',
                            '// Stored signed E4M3 bytes remain unchanged in the weight bank.')
    source = source.replace(ast.get_source_segment(source, node), '_pf_GROUPED_SRC = ' + repr(native), 1)
    assert '_pf_encode_sf_fast' not in source and '_pf_encode_scales' not in source
    return source
