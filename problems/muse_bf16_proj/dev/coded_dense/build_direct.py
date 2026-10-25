"""Reconstruct actual BF16 exponents in timed decoders, removing per-output FP32 scaling."""
import ast
import re
from pathlib import Path
import runpy
H=Path(__file__).resolve().parent
runpy.run_path(str(H/'build_cpall.py'))
runpy.run_path(str(H/'build_wg.py'))

def constant(src,name):
 return next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)

def replace1(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)

def direct_cp(src):
 cuda=constant(src,'CD_CUDA')
 cuda=replace1(cuda,'uint32_t nw, uint32_t (&a)[4])','uint32_t nw, uint32_t (&a)[4], uint32_t origin)')
 cuda=replace1(cuda,'(nw & 0x87878787u) | 0x38383838u','(nw & 0x87878787u) + origin')
 cuda=replace1(cuda,'(__funnelshift_l(nw, nw, 4) & 0x87878787u) | 0x38383838u','(__funnelshift_l(nw, nw, 4) & 0x87878787u) + origin')
 cuda=cuda.replace('const float* __restrict__ scales','const uint32_t* __restrict__ scales')
 cuda=cuda.replace('const float* sc','const uint32_t* sc').replace('(const float*)sc','(const uint32_t*)sc')
 cuda=replace1(cuda,'float sc[RT];','uint32_t sc[RT];')
 cuda=replace1(cuda,'ld_f32(scales + r0 + rt)','ld_i32((const int*)(scales + r0 + rt))')
 cuda=replace1(cuda,'word(buf[p][rt][2], j), a);','word(buf[p][rt][2], j), a, sc[rt]);')
 cuda=replace1(cuda,'    sc[rt] = ld_f32(scales + r0 + rt);','    sc[rt] = ld_i32((const int*)(scales + r0 + rt));') if '    sc[rt] = ld_f32(scales + r0 + rt);' in cuda else cuda
 cuda=replace1(cuda,'#pragma unroll\n    for (int nt = 0; nt < NT; ++nt)\n#pragma unroll\n      for (int i = 0; i < 4; ++i) acc[rt][nt][i] *= sc[rt];','')
 cuda=replace1(cuda,'sc[rt] * 3.0517578125e-05f','__uint_as_float(sc[rt] << 24)')
 cuda=replace1(cuda,'const float sc = scales[r];','const uint32_t sc = scales[r];')
 cuda=replace1(cuda,'word(cur[2], j), a);','word(cur[2], j), a, sc);')
 cuda=replace1(cuda,'const float w = __uint_as_float(bits << 16) * sc;\n        const uint32_t wb = __float_as_uint(w) >> 16;  // exact: w is a normal bf16','const uint32_t wb = bits;')
 tree=ast.parse(src);lines=src.splitlines(True)
 # Replace only the literal CUDA source, leaving all Python encoder code readable for audit.
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CD_CUDA')
 src=''.join(lines[:node.lineno-1])+'CD_CUDA = '+repr(cuda)+'\n'+''.join(lines[node.end_lineno:])
 src=replace1(src,'base = win.argmax(1).to(torch.int32) + 1','base = win[:, 1::2].argmax(1).to(torch.int32) * 2 + 2')
 src=replace1(src,'scales = torch.pow(2.0, (base - 112).double()).float()','origin = ((base >> 1) * 0x01010101).to(torch.int32)')
 src=replace1(src,'scales=scales.contiguous()','origin=origin.contiguous()')
 src=src.replace("c['scales']", "c['origin']")
 return src

cp_payload=(H/'cpall_lab.py').read_text()
cp=constant(cp_payload,'SOURCES')['cpall']
cp=direct_cp(cp)
wg=(H/'wg.py').read_text().replace('from coded import CD_CUDA, cd_encode, cd_decode\n','')
k=(H/'wg_kernel.cu').read_text().replace('const float* __restrict__ scales','const uint32_t* __restrict__ scales')
k=replace1(k,'const float sc=ld_f32(scales+r);','const uint32_t sc=ld_i32((const int*)(scales+r));')
k=replace1(k,'word(buf[p][2],j),A[z][j]);','word(buf[p][2],j),A[z][j],sc);')
k=replace1(k,'#pragma unroll\n  for(int i=0;i<MT/2;++i)acc[i]*=sc;','')
k=replace1(k,'sc*3.0517578125e-05f','__uint_as_float(sc<<24)')
wg=wg.replace('WG_KERNEL = None','WG_KERNEL = '+repr(k))
wg=wg.replace('const float*sc','const uint32_t*sc').replace('(const float*)sc','(const uint32_t*)sc')
wg=wg.replace("c['scales']", "c['origin']")
(H/'direct_body.py').write_text(cp+'\n'+wg)
print(H/'direct_body.py')

old_cp=constant(cp_payload,'SOURCES')['cpall']
old_wg=(H/'wg_frozen.py').read_text().replace('from coded import CD_CUDA, cd_encode, cd_decode\n','')
pre=(H/'lab_body.py').read_text().split('\ndef lab(',1)[0].replace('__CL_VS__','(3,)').replace('__CL_SHAPES__','((8,1,2),)')
(H/'direct_lab.py').write_text('import torch\nimport torch.nn.functional as F\nimport triton\nimport triton.language as tl\n'+pre+'\nSOURCES = '+repr({'old':old_cp+'\n'+old_wg,'direct':cp+'\n'+wg})+'\n'+(H/'direct_lab.py.inc').read_text())
