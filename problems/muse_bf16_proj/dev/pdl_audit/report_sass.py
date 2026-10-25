"""Map SASS load PCs to source lines, keeping pre/post-wait evidence reviewable."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent


def annotate(label):
    folder=HERE/'native'/label;records=[]
    natives=json.loads((HERE/(label+'_native.json')).read_text())
    for native in natives:
        name=native['name'];source=(folder/(name+'.cu')).read_text().splitlines()
        native['sass_sha256']=hashlib.sha256((folder/(name+'.sass')).read_bytes()).hexdigest()
        native['resource_usage']=(folder/(name+'.resources')).read_text()
        text='\n'.join(p.read_text() for p in (folder/(name+'_elf')).glob('*.sass'))
        for kernel in native['kernels']:
            symbol=kernel['symbol']
            section=re.search(r'//-+ \.text\.'+re.escape(symbol)+r'\s+-+\n(.*?)(?=\n//-+ \.|\Z)',text,re.S)
            assert section,(name,symbol)
            fields=[];file=None;line=None;wait_seen=False
            for s in section[1].splitlines():
                m=re.search(r'//## File "([^"]+)", line (\d+)',s)
                if m:file,line=m[1],int(m[2])
                if 'ACQBULK' in s:wait_seen=True
                if re.search(r'\b(?:LDG|ULDG|LDGSTS|CCTL)\b',s) and re.search(r'/\*[0-9a-f]+\*/',s):
                    fields.append(dict(after_wait=wait_seen,line=line,file=Path(file).name if file else None,sass=s.strip(),
                        source=source[line-1].strip() if file and Path(file).name==name+'.cu' and line and line<=len(source) else None))
            demangled=subprocess.run(['c++filt',symbol],text=True,capture_output=True,check=True).stdout.strip()
            records.append(dict(source=name,symbol=demangled,waits=kernel['waits'],loads=fields))
    (HERE/(label+'_native.json')).write_text(json.dumps(natives,indent=2)+'\n')
    # Keep the complete source-line inventory separate from the concise kernel audit.
    (HERE/(label+'_load_lines.json')).write_text(json.dumps(records,indent=2)+'\n')
    print(label,len(records),'native PDL specializations')
    for record in records:
        if record['source'] in ('CUDA_SRC','VQ8_CUDA','PARTIAL_CUDA','GLUE_CL_CUDA','SMALLM_VAR_CUDA'):
            pre=[r for r in record['loads'] if not r['after_wait']]
            after={r['source'] for r in record['loads'] if r['after_wait'] and r['source'] and any(x in r['source'] for x in ('scale','prew','postw','alpha','beta','bias','lstart','p.loc','inv_freq','positions','gs13[','gs2['))}
            print(record['source'],record['symbol'][:85],len(pre),'early loads; late fields',sorted(after),flush=True)


if __name__=='__main__':
    for label in ('baseline','fixed'):annotate(label)
