from pathlib import Path
import runpy
p=Path(__file__).resolve();m=runpy.run_path(str(p.with_name('build_xqa.py')))
c=m['cuda']
c=c.replace('namespace {','namespace {\nconstexpr int KT=128,KB=KT*64;',1)
c=c.replace('i<2048','i<KT*32')
c=c.replace('float s[4][2][4]','float s[KT/16][2][4]')
c=c.replace('kt<4','kt<KT/16').replace('kt<2','kt<KT/32')
c=c.replace('sk+ST*4096','sk+ST*KB').replace('sv+ST*4096','sv+ST*KB')
c=c.replace('nv=nk+4096;uint8_t* vh=nv+4096','nv=nk+KB;uint8_t* vh=nv+KB')
c=c.replace('+63)&~63','+KT-1)&~(KT-1)').replace('(end-begin+63)/64','(end-begin+KT-1)/KT')
c=c.replace('rows[ST][2]','rows[ST][KT/32]').replace('rr<2','rr<KT/32')
c=c.replace('tileid*64','tileid*KT').replace('slot*4096','slot*KB')
c=c.replace('i<8192/16','i<2*KB/16')
c=c.replace('(i%ST)*4096','(i%ST)*KB').replace('begin+i*64','begin+i*KT')
c=c.replace('ST*8192+12288+8192','ST*2*KB+4096+4*KB')
# Only the three-stage family fits the local shared-memory capacity. It keeps
# 32 KiB of raw gathered KV per CTA in flight on Hopper.
c=c.replace('if(stages==5){if(p8)go<5,true>(p,s);else go<5,false>(p,s);}else{if(p8)go<3,true>(p,s);else go<3,false>(p,s);}', 'if(p8)go<3,true>(p,s);else go<3,false>(p,s);')
py=m['py'].replace('X_STAGES = 5','X_STAGES = 3').replace('X_P8 = False','X_P8 = True')
py=py.replace("name=f'muse_xqa_ptranspose_sm{arch}'", "name=f'muse_xqa_k128_sm{arch}'")
py=py.replace('(maxkeys+63)//64','(maxkeys+127)//128')
py=py.replace("window=cfg['window']; maxpages", "window=cfg['window']; maxpages")
py=py.replace("return _xqa_verify(data)","global X_WAVES\n        X_WAVES=1 if data['config']['window'] else 4\n        return _xqa_verify(data)")
root=m['root']
if __name__ == '__main__':
    # The four-warp source is retained for reproduction, never admitted as a
    # candidate: it spills on sm_90a. build_xqa_w8.py builds the replacement.
    (root/'dev/xqa_k128.cu').write_text(c)
