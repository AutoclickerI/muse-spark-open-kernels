"""Render reset-dependent admission packets in seconds; no unselected fallback kernels."""
import argparse,ast,base64,hashlib,json,lzma,re
from pathlib import Path
import build_packet,kernels
H=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('dsm','dsm_recheck','global_confirm','history','selected'))
    ap.add_argument('--selection',required=True,type=json.loads,help='DSM {batch:cluster}; history {"qt":1|2|4}; selected {"batch,q_len":[kind,qt,splits]}')
    a=ap.parse_args()
    # Rebuild the checker/header from tracked inputs. An old generated body
    # must not silently retain a retired tolerance margin.
    build_packet.main()
    body=(H/'out/body.py').read_text()
    cut=body.index('SEEDS=tuple(');body=body[:cut]+(H/'lab_body.inc').read_text()
    body=body.replace('SEEDS=tuple(4194319+104729*i for i in range(16))','SEEDS=tuple(7101149+104729*i for i in range(24))')
    if a.mode in ('dsm','dsm_recheck'):
        choice={int(b):int(c) for b,c in a.selection.items()}
        assert choice and all(b in (1,16,32) and c in (1,2,4,8,16) for b,c in choice.items())
        arms={}
        for b,c in choice.items():
            qt=4 if b==1 else 1
            arms[(b,8)]={'RESET':('dsm_old',qt,c),'COOP':('dsm',qt,c),
                         'HALF':('dsm',qt,max(1,c//2)),'DOUBLE':('dsm',qt,min(16,c*2))}
        cases=tuple(arms)
        if a.mode=='dsm_recheck':
            assert set(choice)=={1},'recheck only the unresolved verify1 control'
            body=body.replace('SEEDS=tuple(7101149+104729*i for i in range(24))',
                              'SEEDS=tuple(12100117+104729*i for i in range(35))')
            body=body.replace('ROUNDS=12','ROUNDS=14')
            body=body.replace("variants={'A':host,'A2':host,", "variants={'A':host,'A2':host,'A3':host,")
    elif a.mode=='global_confirm':
        assert not a.selection,'global confirmation has a receipt-selected fixed matrix'
        # Use the exact native-gated global shader, not a rebuilt variant.
        packet=(H.parents[3]/'runs/muse_attention_decode/20261026T015049Z_check_bb7239afc1b3487a.py').read_text()
        payload=next(ast.literal_eval(n.value)[0][2] for n in ast.parse(packet).body
            if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets))
        blob=next(n.args[0].value for n in ast.walk(ast.parse(payload)) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode' and isinstance(n.args[0],ast.Constant))
        native=lzma.decompress(base64.b85decode(blob)).decode()
        cuda=next(ast.literal_eval(n.value) for n in ast.parse(native).body
            if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='cs_CUDA')
        body+='\nKERNELS["global"]='+repr(cuda)+'\n'
        body+='\nKERNELS["last"]='+repr(kernels.build_last.source())+'\n'
        arms={}
        for b,qt,ns in ((1,4,16),(16,1,8),(32,1,4)):
            arms[(b,1)]={'GLOBAL':('global',qt,ns),'ALIAS':('global',qt,ns),
                         'LAST':('last',qt,ns),'PP':('partial',qt,ns)}
        cases=tuple(arms)
        body=body.replace('SEEDS=tuple(7101149+104729*i for i in range(24))',
                          'SEEDS=tuple(15100087+104729*i for i in range(35))')
        body=body.replace('ROUNDS=12','ROUNDS=14')
        body=body.replace("variants={'A':host,'A2':host,", "variants={'A':host,'A2':host,'A3':host,")
        body=body.replace("        return apply_arm(dict(data,config=cfg),qkv,arm).view(-1,host.Q_SIZE)",
                          "        if arm[0]=='partial':return host._layer_partial_port(qkv,data)\n        return apply_arm(dict(data,config=cfg),qkv,arm).view(-1,host.Q_SIZE)")
        body=body.replace("for n,(k,qt,ns) in arms.items()}","for n,(k,qt,ns) in arms.items() if k!='partial'}")
        body=body.replace("got=apply_arm(dict(data,config=cfg),qkv,a).view(-1,host.Q_SIZE)",
                          "got=host._layer_partial_port(qkv,data) if a[0]=='partial' else apply_arm(dict(data,config=cfg),qkv,a).view(-1,host.Q_SIZE)")
        body=body.replace("if n in arms:assert len(profiles[n]['nodes'])==1", "if n in arms and arms[n][0]!='partial':assert len(profiles[n]['nodes'])==1")
    elif a.mode=='history':
        qt=int(a.selection['qt']);assert qt in (1,2,4)
        # Reuse the exact reset early-history shader for a held-out confirmation,
        # and test its held-out whole-layer result.
        import sys
        sys.path.insert(0,str(H.parent/'early_history'));import history_build
        oldcuda=history_build.source()
        assert hashlib.sha256(oldcuda.encode()).hexdigest()=='0e07854b0a329f411d21ae9c18583646dccaff456a7fbdfe446608e14de58d58'
        body+='\nKERNELS["early"]='+repr(oldcuda)+'\n'
        lastcuda=history_build.build_last.source()
        assert hashlib.sha256(lastcuda.encode()).hexdigest()=='3815beabedbbb5773536dd33b765fe223821624ed4eadf829b122973028ee85a'
        body+='\nKERNELS["last"]='+repr(lastcuda)+'\n'
        arms={(1,1):{'EARLY':('early',qt,16),'OLD4':('last',4,16)}};cases=tuple(arms)
    else:
        # Confirm only winners actually selected by the native receipt.
        # This uses the same exact-S14 capture/checker and held-out seeds.
        arms={tuple(map(int,k.split(','))):{'SELECTED':tuple(v)} for k,v in a.selection.items()}
        assert arms and all(k in ((16,1),(32,1),(1,8)) for k in arms)
        assert all(v[0] in ('last','early','dsm','dsm_old') and v[1] in (1,2,4,8)
                   and 1<=v[2]<=32 for row in arms.values() for v in row.values())
        cases=tuple(arms)
    # Replace only metadata constants, preserving shared capture, TASK gates,
    # cold inputs and all fixed public-checker math.
    body+='\nARMS='+repr(arms)+'\nCASES='+repr(cases)+'\n'
    used={v[0] for row in arms.values() for v in row.values()}
    lines=body.splitlines(True)
    for n in reversed(ast.parse(body).body):
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='KERNELS':
            value=ast.literal_eval(n.value);value={k:v for k,v in value.items() if k in used}
            lines[n.lineno-1:n.end_lineno]=['KERNELS='+repr(value)+'\n']
        elif isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) and getattr(n.targets[0].value,'id','')=='KERNELS':
            key=ast.literal_eval(n.targets[0].slice)
            if key not in used:lines[n.lineno-1:n.end_lineno]=[]
    body=''.join(lines)
    body=body.replace("END shapes=%d/3 seeds=%d",'END shapes=%d/'+str(len(cases))+' seeds=%d')
    body=body.replace('META 16 fresh seeds x12 rotating cold READ rounds;',
        'META '+('35 fresh seeds x14' if a.mode in ('dsm_recheck','global_confirm') else '24 fresh seeds x12')+
        ' rotating cold READ rounds; fresh graph pools/capture ownership per seed;')
    if a.mode=='global_confirm':
        body=body.replace('last-CTA decode16/32+verify1, early-history and DSM in one paired batch.',
                          'native-gated global128/identical alias/last/partial-port at decode1/16/32 sliding.')
        body=body.replace('one attention node.', 'global/last one launch; PP is the existing partial-port control.')
    # h200_lab's generic wrapper retains 9000 characters, less than the
    # original round2 board's 15000. Keep the complete archive within it.
    body=body.replace('>13500:', '>7000:').replace("<14500,'round2 receipt would truncate'", "<8800,'follow-up receipt would truncate'")
    compile(body,'round2_followup','exec');packet=build_packet.boot(body)
    packet+='\n_P5_FOLLOWUP_LAB=lab\ndef lab(deadline,weights,state=None):\n    return _P5_FOLLOWUP_LAB(deadline,weights,state)\n'
    p=H/'out'/('followup_'+a.mode+'.py');p.write_text(packet)
    print(p,hashlib.sha256(packet.encode()).hexdigest(),'cases',cases)
if __name__=='__main__':main()
