"""Inventory closed-recipe primary receipts and the actual old gate predicates."""
import ast,base64,hashlib,json,lzma,math,re,statistics
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]


def payloads(source):
    todo=[('submission',source)];seen=set();out=[]
    while todo:
        label,text=todo.pop();key=hashlib.sha256(text.encode()).hexdigest()
        if key in seen:continue
        seen.add(key);out.append((label,text))
        try:tree=ast.parse(text)
        except (SyntaxError,ValueError):continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('b85decode','b64decode') and node.args:
                try:
                    value=ast.literal_eval(node.args[0]);data=getattr(base64,node.func.attr)(value);expanded=lzma.decompress(data).decode()
                except (ValueError,TypeError,lzma.LZMAError,UnicodeDecodeError):continue
                if 'def ' in expanded:todo.append((label+':compressed',expanded))
            if isinstance(node,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in node.targets):
                try:items=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                todo.extend((name,body) for name,sha,body in items)
    return out


def archives(log):
    out=[]
    for line in log.splitlines():
        match=re.match(r'^([A-Z][A-Z0-9_]+) (.+)$',line)
        if not match:continue
        tag,body=match.groups();data=None
        try:
            if any(x in tag for x in ('XZ','Z85')):data=json.loads(lzma.decompress(base64.b85decode(body)))
            elif body.startswith(('{','[')):data=json.loads(body)
        except (ValueError,TypeError,lzma.LZMAError,UnicodeDecodeError):pass
        if data is not None:out.append((tag,data))
    return out


def arrays(data,path=''):
    out=[]
    if isinstance(data,dict):
        for k,v in data.items():out+=arrays(v,path+'/'+str(k))
    elif isinstance(data,list):
        if len(data)>=3 and all(isinstance(v,(float,int)) and not isinstance(v,bool) for v in data):
            out.append(dict(path=path,n=len(data),minimum=min(data),maximum=max(data)))
        elif data and isinstance(data[0],(dict,list)):
            for i,v in enumerate(data[:3]):out+=arrays(v,path+'/'+str(i))
    return out


def main():
    text=(R/'docs/FINDINGS.md').read_text();section=text.split('## Closed recipes\n',1)[1].split('\n## Failures',1)[0]
    rows=[]
    for line in section.splitlines():
        if not line.startswith('| ') or 'Exact recipe' in line:continue
        parts=[x.strip() for x in line.strip('|').split('|')]
        if len(parts)<5:continue
        rows.append(dict(recipe=parts[0],old_result=parts[1],mechanism=parts[2],leaves_open=parts[3],prefixes=re.findall(r'\b[0-9a-f]{8}\b',parts[-1])))
    files={};receipts={};prefixes={p for row in rows for p in row['prefixes']}
    for path in (R/'runs').glob('*/*.json'):
        if path.parent.name in ('pending','waiting','sent','detached'):continue
        if any('_'+p in path.name for p in prefixes):
            data=json.loads(path.read_text())
            if data.get('created_at',0) and data['created_at']<1792681200:continue
            receipts[data['id']]=(path,data)
    generated=H/'generated';generated.mkdir(exist_ok=True)
    for job,(path,data) in receipts.items():
        log=data.get('result',{}).get('log_tail','');source=path.with_suffix('.py');chunks=payloads(source.read_text()) if source.exists() else []
        decoded=archives(log);evidence=[]
        for label,body in chunks:
            lines=body.splitlines()
            for i,line in enumerate(lines):
                if len(line)>350:continue
                if re.search(r'def \w*(gate|agree|check)\w*\(|torch\.equal|if .*\b(?:err|rel|l2)\b.*[<>]|all\(.+(wins|ratio)|ids.*(equal|!=)|[.]003\b',line,re.I):
                    evidence.append(dict(source=label,line=i+1,text='\n'.join(lines[max(0,i-1):i+3])[:850]))
        (generated/(job+'.log')).write_text(log)
        groups={}
        for tag,value in decoded:groups.setdefault(tag,[]).append(value)
        for tag,values in groups.items():(generated/(job+'_'+tag+'.json')).write_text(json.dumps(values[0] if len(values)==1 else values,indent=2)+'\n')
        files[job]=dict(path=str(path.relative_to(R)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
            created_at=data.get('created_at'),official_host_pass=data.get('result',{}).get('passed'),
            log_chars=len(log),tags=[tag for tag,value in decoded],arrays={tag:arrays(values[0] if len(values)==1 else values)[:80] for tag,values in groups.items()},
            predicates=evidence[:45],TASK_checker_in_lab=any('check_implementation' in body for _,body in chunks[1:]),
            errors=[line[:240] for line in log.splitlines() if re.match(r'^(FAIL|ERROR|LAB_ERROR|LAB_TRUNCATED)',line)],
            log_preview=[line[:220] for line in log.splitlines() if len(line)<250 and re.match(r'^(META|PAIR|END|RESULT|R |AA|CASE|GATE)',line)][:65])
    for row in rows:row['jobs']=[job for job in files if job[:8] in row['prefixes']]
    report=dict(since='2026-10-23',rows=rows,receipts=files,method='Primary immutable runs/ receipts; archived predicates and raw-array schemas. No CI reconstructed from absent covariance.')
    (H/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Closed rows',len(rows),'primary receipts',len(files),'archives',sum(len(r['tags']) for r in files.values()))
    for job,r in files.items():
        if r['tags'] or r['errors']:print(job[:8],r['tags'],r['errors'][:2])
if __name__=='__main__':main()
