"""Pack exact sources and independently composed off arms into one reset lab."""
import ast
import base64
import difflib
import hashlib
import importlib.util
import json
import lzma
import sys
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path.insert(0,str(H))
import build


def main():
    source,meta=build.compose()
    assert (H.parents[1]/'cand_r3_best7_s15.py').read_text()==source
    variants={}
    for name in build.PIECES:
        text,_=build.compose(without=(name,))
        variants[name]=dict(source=text,active_tokens=build.PIECES[name])
    for role in ('qkv','o','unpooler'):
        text,_=build.compose(without_role=role)
        variants[role]=dict(source=text,active_tokens=sorted({t for t,r in json.loads((H/'proj/selection.json').read_text())['selected']
            if r==role or role=='qkv' and r in ('qkv3','qkv_gate')}))
    variants['S14']=dict(source=build.BASE.read_text(),active_tokens=[1,8,16,32,128,256])
    packed={};host_lines=source.splitlines(True)
    for name,arm in variants.items():
        text=arm.pop('source');lines=text.splitlines(True)
        edits=[(a,b,''.join(lines[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,host_lines,lines,autojunk=False).get_opcodes() if op!='equal']
        restored=list(host_lines)
        for a,b,replacement in reversed(edits):restored[a:b]=[replacement]
        assert ''.join(restored)==text
        packed[name]=dict(arm,edits=edits,sha256=hashlib.sha256(text.encode()).hexdigest())
    helpers=(H.parent/'s10/ablate_lab.py.inc').read_text();nodes=[]
    wanted={'ab_label','ab_namespace','ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','ab_summary','AB_CASES','AB_LOADS'}
    for n in ast.parse(helpers).body:
        name=n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
        if name in wanted:nodes.append(ast.get_source_segment(helpers,n))
    bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,variants=packed),separators=(',',':')).encode(),preset=9)).decode()
    body='import torch,types,time,statistics,math\n'+'\n\n'.join(nodes)+'\nAB_SEEDS=tuple(3121907+7919*i for i in range(16))\n'
    body+='import base64,lzma,json\n_S15_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\n'
    body+=(H/'ab_lab.py.inc').read_text()
    encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    digest=hashlib.sha256(body.encode()).hexdigest()
    bootstrap='import base64,lzma,linecache,hashlib\n'
    bootstrap+='_s15_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    bootstrap+='assert hashlib.sha256(_s15_body.encode()).hexdigest()=='+repr(digest)+'\n'
    bootstrap+='_s15_file="<s15ab-body-'+digest[:12]+'>"\nlinecache.cache[_s15_file]=(len(_s15_body),None,_s15_body.splitlines(True),_s15_file)\n'
    bootstrap+='exec(compile(_s15_body,_s15_file,"exec"),globals())\n_s15_lab=lab\ndef lab(deadline,weights,state):\n    return _s15_lab(deadline,weights,state)\n'
    payload=H/'s15ab_payload.py';payload.write_text(bootstrap)
    # Compress the outer candidate as well. Its expanded source is audited separately;
    # official TASK correctness still runs the exact candidate. Keep the MIT notice plain.
    notice='\n'.join('# '+line for line in ('Nathen Fernandes; MIT\n'+(H/'proj/LICENSE.NathenFernandes').read_text()).splitlines())
    packed_source=base64.b85encode(lzma.compress(source.encode(),preset=9)).decode()
    outer='# okc: problem=muse_decode_layer\n'+notice+'\nimport base64,lzma,linecache,hashlib,torch\n'
    outer+='_s15_source=lzma.decompress(base64.b85decode('+repr(packed_source)+')).decode()\n'
    outer+='assert hashlib.sha256(_s15_source.encode()).hexdigest()=='+repr(meta['sha256'])+'\n'
    outer+='_s15_path="<s15-expanded-'+meta['sha256'][:12]+'>"\nlinecache.cache[_s15_path]=(len(_s15_source),None,_s15_source.splitlines(True),_s15_path)\n'
    outer+='exec(compile(_s15_source,_s15_path,"exec"),globals())\n_s15_prepare=globals()["prepare"]\ndef prepare(weights):\n    return _s15_prepare(weights)\n'
    host=H/'s15ab_host.py';host.write_text(outer)
    meta.update(payload=str(payload.relative_to(R)),payload_sha256=build.sha(payload),
                host_file=str(host.relative_to(R)),host_file_sha256=build.sha(host),arms=packed,body_sha256=digest)
    (H/'ab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('host',len(outer),'payload',len(bootstrap),'expanded',len(source))


if __name__=='__main__':main()
