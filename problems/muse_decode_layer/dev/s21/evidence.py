"""Losslessly deduplicate equal quantized sample vectors; never omit a seed or arm."""
def s21_pack_samples(loads,names,scale):
    vectors=[];seen={};aliases={}
    for label,cell in loads.items():
        aliases[label]={}
        for name in names:
            values=tuple(round(row[name]*scale)-scale for row in cell['samples'])
            if values not in seen:
                seen[values]=len(vectors);vectors.append(list(values))
            aliases[label][name]=seen[values]
    return dict(sample_encoding='deduplicated-i32-vectors-v1',sample_vectors=vectors,sample_aliases=aliases)

def s21_decode_samples(raw):
    assert raw['sample_encoding']=='deduplicated-i32-vectors-v1'
    vectors=raw.pop('sample_vectors');aliases=raw.pop('sample_aliases');names=raw['sample_arms']
    assert set(aliases)==set(raw['sample_loads'])
    for label in raw['sample_loads']:
        n=raw['loads'][label]['n'];indices=aliases[label];assert set(indices)==set(names)
        assert all(isinstance(i,int) and 0<=i<len(vectors) and len(vectors[i])==n for i in indices.values())
        raw['loads'][label]['samples']=[{name:1+vectors[indices[name]][i]/raw['sample_scale'] for name in names} for i in range(n)]
    return raw
