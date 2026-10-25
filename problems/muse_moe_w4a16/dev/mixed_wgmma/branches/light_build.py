"""Decode 9-16-token light experts once with N16, beside NT1<=8 and heavy.

The roof diagnostic found NT1 near the roof for its repeated bytes. This tests
removing the second NVFP4 read without padding more tokens or changing formats.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H),str(H.parent),str(R/'tools')]
import build
import patch_s12_permw
from build_prefill_lab import subset


def once(s,a,b):
    assert s.count(a)==1,a[:100];return s.replace(a,b)


def value(s,key):
    return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key)


def main():
    s=(R/'problems/muse_decode_layer/cand_r3_best7_s12_final.py').read_text()
    assert hashlib.sha256(s.encode()).hexdigest()=='240248b2c27c890bf5055a77e87d2fc9c4429b747a04d714e51d5f67e936bb3e'
    perm=patch_s12_permw.patch(s);c0=value(s,'NF_CUDA');c1=value(perm,'NF_CUDA')
    cu=once(c1,'F(26,32,true,7,0,0,0) F(21,32,false,6,0,0,0)',
            'F(26,32,true,7,0,0,0) F(21,32,false,6,0,0,0) F(70,16,true,7,1,8,16) F(71,16,false,6,1,8,16)')
    cu=once(cu,'set.template operator()<true,true,16>();','set.template operator()<true,true,16>();set.template operator()<true,true,8>();')
    cu=once(cu,'BR_REQUIRE(first?thr==16:(thr==16||thr==8));','BR_REQUIRE(first?(thr==16||thr==8):(thr==16||thr==8));')
    cu=once(cu,'if(first)launch.template operator()<true,true,16>();else if(thr==16)',
            'if(first&&thr==8)launch.template operator()<true,true,8>();else if(first)launch.template operator()<true,true,16>();else if(thr==16)')
    funcs=build.strip_imports(subset(perm,('nf_plan','nf_call','nf_ilv','nf_wide','nf_perm')))
    node=next(n for n in ast.parse(perm).body if isinstance(n,ast.FunctionDef) and n.name=='nf_plan')
    nl=(ast.get_source_segment(perm,node)+'\n').replace('def nf_plan(data,base,native,streams):','def nf_light_plan(data,base,native,streams,light_grid):')
    nl=once(nl,'    rp=[t.data_ptr()',
            '    l13=list(h13);l2=list(h2);l13[7]=tasks[2];l2[7]=tasks[2]\n    pl13=[t.data_ptr() for t in l13];pl2=[t.data_ptr() for t in l2]\n    rp=[t.data_ptr()')
    nl=once(nl,'owners=(h13,h2,nv,ids,tw,pos,tasks)','owners=(h13,h2,nv,ids,tw,pos,tasks,l13,l2)')
    nl=once(nl,'side,fork,join=streams;grid=', 'side,fork,join,light,ljoin=streams;grid=')
    nl=once(nl,'fork.record(parent);side.wait_event(fork)','fork.record(parent);side.wait_event(fork);light.wait_event(fork)')
    nl=once(nl,'        mod.pmz_run(pn,1,16,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,16,m,stages,grids[0],parent.cuda_stream)',
            '''        with torch.cuda.stream(light):
            mod.br_run(pl13,70,sms*light_grid,m,light.cuda_stream);mod.br_run(pl2,71,sms*light_grid,m,light.cuda_stream);ljoin.record()
        mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)''')
    nl=once(nl,'parent.wait_event(join)','parent.wait_event(join);parent.wait_event(ljoin)')
    body=build.helpers()+'\n'+build.IMPORTS+'\n'+build.strip_imports(build.HH_CELL.read_text())+'\n'+funcs+'\n'+nl+'\n'
    body+=f'OP_CUDA={[c0,c1,cu,cu]!r}\nOP_CPP={value(s,"NF_CPP")!r}\nNF_OPTS={{}}\n'
    runtime=(H/'order_runtime.py.inc').read_text()
    runtime=once(runtime,"OP_NAMES=('S12','PERMW','ORDER','PERMW_ORDER')","OP_NAMES=('S12','PERMW','N16G2','N16G4')")
    runtime=once(runtime,"return dict(bases=(base,perm,base,perm),native=native,\n                streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()) for _ in OP_NAMES])",
                 "return dict(bases=(base,perm,perm,perm),native=native,\n                streams=[(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()) if a<2 else (torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) for a in range(4)])")
    runtime=once(runtime,"    return nf_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm])",
                 "    if arm>=2:return nf_light_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm],2 if arm==2 else 4)\n    return nf_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm])")
    runtime=runtime.replace('META exact S12 final 240248b2 vs PERMW, ORDER, PERMW_ORDER; same N32 tile math, geometry and global weight reads. ORDER changes only (expert,token-group,channel) to (expert,channel,token-group), shortening inter-CTA weight reuse distance.',
                            'META S12 final vs PERMW vs N16G2/G4: NT1 only <=8 tokens; interleaved N16 WGMMA handles 9-16-token experts once on a third stream, heavy N32 unchanged. N16 persistent grid=2/4 CTAs per SM. Same work/precision; fewer repeated weight bytes.')
    body+=runtime;compile(body,'<N16 light lab>','exec');(H/'light_payload.py').write_text(body)
    (H/'cu_light.cu').write_text(cu)
    manifest=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),light_cuda_sha256=hashlib.sha256(cu.encode()).hexdigest())
    (H/'light_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)


if __name__=='__main__':main()
