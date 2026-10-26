"""Select layer replacements by six-load geomean; promotion needs fresh pairs."""
import base64,copy,json,lzma,math,re,statistics
from pathlib import Path
import decide,patches
from score_gate import T95
H=Path(__file__).resolve().parent;R=H.parents[3]


def parsed(log):
    rows=[json.loads(s) for s in re.findall(r'^RESULT (\{.*\})$',log,re.M)]
    rows += [json.loads(lzma.decompress(base64.b85decode(s))) for s in re.findall(r'^RESULT_XZ85 (\S+)$',log,re.M)]
    assert len(rows)==1,'missing/truncated layer result'
    return rows[0]


def nominate(files):
    recovery=json.loads((H/'layer_recovery.json').read_text());selected=[];evidence=[];receipts={}
    for name,entry in recovery.items():
        first=len(selected);first_evidence=len(evidence)
        try:
            receipt=decide.read_receipt(entry,files);raw=parsed(receipt['log']);receipts[name]=receipt
            assert raw['complete'] and not raw.get('errors') and not raw.get('failures')
            if name=='nearby':
                # Identical dispatch at the scored shapes: nearby measurements
                # do not establish a six-load gain. Keep them as diagnostics.
                assert raw['shape_count']==65 and len(raw['rows'])==65
                evidence.append(dict(arm='layer_nearby',wins=False,R=1.,CI95=None,
                                     reason='scored dispatch identical; nearby-only evidence',job=receipt['job']))
                continue
            assert raw['seeds']==8 and len(raw['shapes'])==12 and len(raw['loads'])==6
            assert all(row['n']==8 for row in raw['shapes'].values())
            host='E' if name=='release' else 'L'
            groups=('attention','moe','projections') if name=='release' else ('o','pool128','unpool128','v1full','v32slide')
            for group in groups:
                arm='layer_'+name+'_'+group;spec=patches.registry()[arm]
                cells=raw['loads']
                if name=='library':
                    # These replacements dispatch on projection M or a single
                    # attention shape. Choose at the actual dispatch granularity,
                    # without letting noise on untouched loads select a loser.
                    spec=copy.deepcopy(spec);kept=[];logs=[];errors=[]
                    points=sorted({(b,q) for b,q,w in spec['rows']})
                    shape_evidence=[]
                    for b,q in points:
                        cell=cells[f'{b}x{q}'];off=cell['no_'+group];full=cell[host]
                        ratio=off[0]/full[0]
                        shape_error=T95[7]*(off[1]/off[0]+full[1]/full[0])
                        shape_evidence.append(dict(load=f'{b}x{q}',R=ratio,CI95_approx=[ratio*math.exp(-shape_error),ratio*math.exp(shape_error)],selected=ratio>1))
                        if ratio>1:
                            kept += [row for row in spec['rows'] if tuple(row[:2])==(b,q)]
                            logs.append(math.log(ratio));errors.append(off[1]/off[0]+full[1]/full[0])
                    gain=math.exp(sum(logs)/6);se=sum(errors)/6
                    spec['rows']=kept;wins=bool(kept)
                    # Also expose the all-row aggregate to make the untouched
                    # noise behind the old unpooler nomination reviewable.
                    all_row_R=math.exp(statistics.mean(math.log(c['no_'+group][0]/c[host][0]) for c in cells.values()))
                else:
                    logs=[math.log(c['no_'+group][0]/c[host][0]) for c in cells.values()]
                    gain=math.exp(statistics.mean(logs));wins=gain>1;all_row_R=gain;shape_evidence=[]
                    se=statistics.mean(c['no_'+group][1]/c['no_'+group][0]+c[host][1]/c[host][0] for c in cells.values())
                # Old labs discarded seed covariance. These summary intervals
                # remain descriptive; only a fresh composition lab can promote.
                error=T95[7]*se
                evidence.append(dict(arm=arm,wins=wins,R=gain,all_row_R=all_row_R,CI95=None,
                    CI95_approx=[gain*math.exp(-error),gain*math.exp(error)] if wins or name=='release' else None,
                    shape_selection=shape_evidence,stage_only=True,
                    reason='selection only; shape pieces chosen at dispatch granularity, policies by six-load R; paired samples absent',job=receipt['job']))
                if wins:selected.append(copy.deepcopy(spec))

        except (AssertionError,KeyError,ValueError,StopIteration,lzma.LZMAError,EOFError) as exc:
            del selected[first:];del evidence[first_evidence:]
            evidence.append(dict(batch=name,wins=False,unmeasured=True,reason=type(exc).__name__+':'+str(exc)))
    return selected,evidence,receipts
