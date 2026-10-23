"""Build a 128-key, eight-warp XQA tile without the old 128-key register spill.

Each warp owns eight query rows. QK uses raw FP8 ldmatrix operands. PV uses
exact widened V or raw FP8 ldmatrix.trans plus a packed register transpose.
"""
from pathlib import Path
import runpy

here = Path(__file__).resolve().parent
m = runpy.run_path(str(here / 'build_xqa128.py'))
c, py, root = m['c'], m['py'], m['root']

def replace(old, new, count=None):
    global c
    n = c.count(old)
    assert n > 0 and (count is None or n == count), (old, n, count)
    c = c.replace(old, new)

replace('__launch_bounds__(128,2)', '__launch_bounds__(256,1)', 1)
replace('i+=128', 'i+=256')
replace('rr*32', 'rr*64')
replace('rr<KT/32', 'rr<KT/64')
replace('rows[ST][KT/32]', 'rows[ST][KT/64]')
replace('r<16;r++', 'r<8;r++', 1)
replace('query=w+4*r', 'query=w+8*r', 1)
replace('r<4;r++', 'r<2;r++', 1)
replace('row=w+4*r', 'row=w+8*r', 1)
replace('qb[2][2][2]', 'qb[1][2][2]', 1)
replace('o[4][2][4]', 'o[4][1][4]', 1)
replace('m[2][2],l[2][2]', 'm[1][2],l[1][2]', 1)
replace('s[KT/16][2][4]', 's[KT/16][1][4]', 1)
replace('pb[2][2]', 'pb[1][2]')
replace('nt<2', 'nt<1')
replace('w*16+nt*8', 'w*8+nt*8')
replace('query=w;query<64;query+=4', 'query=w;query<64;query+=8', 1)
replace('<<<dim3(p.ns,p.hkv,p.batch),128,', '<<<dim3(p.ns,p.hkv,p.batch),256,', 1)

start = c.index('   #pragma unroll\n   for(int kt=0;kt<KT/16;kt++){', c.index('float mx=-INFINITY;'))
stop = c.index('   mx=fmaxf(mx,__shfl_xor_sync', start)
masked = c[start:stop]
plain = masked.replace('     if(key>=hi || key>seq+j || (window && key<seq+j-window+1))z=-INFINITY;\n', '')
plain = plain.replace('     int key=kp+16*kt+g+8*rh;\n', '')
c = c[:start] + '''   // The interior of a full/sliding split needs no per-score mask.
   if(kp+KT<=hi && kp+KT<=seq && (!window || kp>=seq+8-window)){
''' + plain + '   }else{\n' + masked + '   }\n' + c[stop:]

old = 'uint32_t a[4]={word(K,kt*16+g,kk*32+4*t),word(K,kt*16+g+8,kk*32+4*t),word(K,kt*16+g,kk*32+16+4*t),word(K,kt*16+g+8,kk*32+16+4*t)};'
replace(old, '''uint32_t a[4];
   uint32_t ka=su(K+off(kt*16+(lane&7)+8*((lane>>3)&1),kk*32+16*(lane>>4)));
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(ka));''', 1)

# A raw 32-key x 16-value FP8 tile is four transposed 8x8 b16 matrices.
# Each b16 packs adjacent value dimensions. Shuffles collect four key bytes
# at one value dimension, matching the FP8 MMA A operand without scalar loads.
begin = c.index('    uint32_t a[4]={0,0,0,0};')
end = c.index('    #pragma unroll\n    for(int nt=0;', begin)
c = c[:begin] + '''    uint32_t r[4],a[4];
    uint32_t va=su(V+off(kt*32+(lane&7)+8*(lane>>3),dt*16));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(va));
    #pragma unroll
    for(int rh=0;rh<2;rh++){
     int src=((g+8*rh)>>1)*4+((2*t)&3);
     uint32_t b[4];
     #pragma unroll
     for(int i=0;i<4;i++){
      uint32_t x=__shfl_sync(0xffffffff,r[i],src);
      uint32_t y=__shfl_sync(0xffffffff,r[i],src+1);
      b[i]=__byte_perm(x,y,(g&1)?0x7531:0x6420);
     }
     a[rh]=t<2?b[0]:b[1];a[rh+2]=t<2?b[2]:b[3];
    }
''' + c[end:]

# Hopper has room for deeper gather rings. The local device can execute the
# three-stage specialization; both use at least 32 KiB of outstanding KV.
replace('if(p8)go<3,true>(p,s);else go<3,false>(p,s);', '''if(stages==5){if(p8)go<5,true>(p,s);else go<5,false>(p,s);}
 else if(stages==4){if(p8)go<4,true>(p,s);else go<4,false>(p,s);}
 else{if(p8)go<3,true>(p,s);else go<3,false>(p,s);}''', 1)

# The eight freshly generated keys only need one K32 FP8 PV tile. Giving this
# tail its own template avoids doing a masked 128-key tile and reserves 4 KiB
# instead of 16 KiB for the two new-cache shared buffers.
replace('DI void widen(', 'template<int TK> DI void widen(', 1)
replace('i<KT*32', 'i<TK*32', 1)
replace('template<bool P8>\nDI void tile(', 'template<int TK,bool P8>\nDI void tile(', 1)
replace('widen(V,vh)', 'widen<TK>(V,vh)', 1)
replace('s[KT/16]', 's[TK/16]', 1)
replace('kt<KT/16', 'kt<TK/16')
replace('kt<KT/32', 'kt<TK/32')
replace('kp+KT<=', 'kp+TK<=')
replace('tile<P8>(st,sk', 'tile<KT,P8>(st,sk', 1)
replace('tile<P8>(st,nk', 'tile<32,P8>(st,nk', 1)
replace('nv=nk+KB;uint8_t* vh=nv+KB', 'nv=nk+2048;uint8_t* vh=nv+2048', 1)
replace('i<2*KB/16', 'i<4096/16', 1)
replace('ST*2*KB+4096+4*KB', 'ST*2*KB+8192+(P8?0:2*KB)', 1)

# Runtime indexing of rows[ST][2] silently creates local-memory traffic even
# when ptxas reports zero register spills. Only the next issued tile's two
# physical rows need registers; prefetch the following mapping during compute.
replace('rows[ST][KT/64]', 'rows[KT/64]', 1)
replace('auto load_rows=[&](int tileid,int slot)', 'auto load_rows=[&](int tileid)', 1)
replace('rows[slot][rr]', 'rows[rr]')
replace(' #pragma unroll\n for(int st=0;st<ST;st++)load_rows(st,st);\n', '', 1)
replace('for(int stage=0;stage<ST-1;stage++){issue(stage,stage);commit();}',
        'for(int stage=0;stage<ST-1;stage++){load_rows(stage);issue(stage,stage);commit();}\n load_rows(ST-1);', 1)
replace('load_rows(i+ST,(i+ST)%ST)', 'load_rows(i+ST)', 1)

py = py.replace('muse_xqa_k128_sm', 'muse_xqa_w8_sm')
py = py.replace('X_P8 = True', 'X_P8 = False')
py = py.replace('X_WAVES = 2', 'X_WAVES = 4')
py = py.replace("X_WAVES=1 if data['config']['window'] else 4",
                "X_WAVES=1 if data['config']['window'] else (8 if data['config']['batch']>=32 else 4)")
for unused in (
    'DI uint32_t word(const uint8_t* v,int r,int c){return *reinterpret_cast<const uint32_t*>(v+off(r,c));}\n',
    'DI uint8_t byte(const uint8_t* v,int r,int c){return v[off(r,c)];}\n',
):
    c = c.replace(unused, '')
text = m['m']['render'](c, py)
start = text.index('"""')
stop = text.index('"""', start + 3) + 3
text = text[:start] + '''"""Muse attention: incumbent q_len=1 plus eight-warp swap-AB q_len=8.

For each KV head, all 64 query rows occupy MMA N and 128 gathered keys occupy
M in 16-key subtiles. Raw FP8 QK uses m16n8k32; FP32 online softmax feeds either
FP16 PV with exactly widened V, or FP8 P/raw V with packed matrix transposes.
A 3/4/5-stage cp.async page-row gather has 32/48/64 KiB outstanding per CTA.
Split-KV CTAs scale to the GPU SM count; the last arriving CTA merges FP32
(max, sum, output), applies RMSNorm and sigmoid gate, and resets its counter.
The eight causal new keys use a separate K32 tile. Every key is attended, with
an exact 2048-token sliding window. No input-derived state survives the call.
"""''' + text[stop:]
(here / 'xqa_w8.cu').write_text(c)
(here / 'xqa_w8_candidate.py').write_text(text)
print('Built', here / 'xqa_w8_candidate.py')
