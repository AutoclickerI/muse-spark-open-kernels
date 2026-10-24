"""Assemble S2 plus receipt-selected persistent-chain rows, then screen the output."""
from pathlib import Path
import argparse,ast,hashlib,json
R=Path(__file__).resolve().parents[4];H=Path(__file__).parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows',type=Path,required=True,help='JSON list of [batch,q_len,window] rows cleared by the H200 receipt')
    ap.add_argument('--receipt',required=True)
    a=ap.parse_args();base=R/'problems/muse_decode_layer/cand_r3_best7_scale_w2fix.py';s=base.read_text()
    manifest=json.loads((H.parent/'s2/manifest.json').read_text())
    assert hashlib.sha256(s.encode()).hexdigest()==manifest['sha256']
    tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='BEST7_DENSE_CHAIN' for t in n.targets))
    table=ast.literal_eval(node.value);rows=[tuple(k) for k in json.loads(a.rows.read_text())]
    if not rows:raise ValueError('receipt must clear at least one row')
    for key in rows:
        if key not in table or key=='other':raise ValueError(key)
        table[key]=True
    s=s.replace(ast.get_source_segment(s,node),'BEST7_DENSE_CHAIN = '+repr(table),1)
    s+='\n\n'+(H/'dense_gate.py.inc').read_text()
    out=base.with_name('cand_r3_best7_s3_chain.py');out.write_text(s)
    result=dict(source=str(out.relative_to(R)),sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=manifest['sha256'],receipt=a.receipt,rows=rows,status='assembled; normal and forced screens still required')
    (H/'dense_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
