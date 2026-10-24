"""M128 scheduling/tiling transform of the shared packed WGMMA producer.

The mainloop, conversion, completion protocol and fused epilogues remain shared
with fused.cu. Only token extent, active MMA width and task traversal change.
"""
from pathlib import Path
import re

CONFIGS = ((16, 256, 4, 1, False), (64, 128, 4, 1, False),
           (64, 128, 4, 1, True), (128, 64, 4, 1, True),
           (128, 128, 2, 1, True), (128, 64, 4, 1, False))


def source():
    text = Path(__file__).resolve().parents[1].joinpath('fused.cu').read_text()
    text = text.replace('namespace mwft {', 'namespace mw128 {').replace('mwft::', 'mw128::')
    text = re.sub(r'\bft_', 'm128_', text)
    # Hopper's N128 accumulator is the same first N16/32/64 register layout.
    marker = '\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16'
    assert marker in text
    text = text.replace(marker, '\n }else if constexpr(NT==64){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16', 1)
    end = '\n }\n#else\n int lane=threadIdx.x&31;'
    regs = ','.join('%'+str(i) for i in range(64))
    outputs = ','.join('"+f"(c['+str(i)+'])' for i in range(64))
    wide = '\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n "{' + regs + '},{%64,%65,%66,%67},%68,p,1,1,0;}"\n :' + outputs + '\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));' + end
    assert end in text
    text = text.replace(end, wide, 1)
    text = text.replace('template<int NT,bool OVERLAP>', 'template<int NT,bool OVERLAP,bool ADAPT>')
    text = text.replace('const __nv_bfloat16*xs){\n#pragma unroll\nfor(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);', '''const __nv_bfloat16*xs,int active){
#pragma unroll
for(int b=0;b<4;++b){
 if constexpr(ADAPT){
  if(active<=16)mma<16>(c,a+b*4,xs,b);
  else if(active<=32)mma<32>(c,a+b*4,xs,b);
  else if(active<=64)mma<64>(c,a+b*4,xs,b);
  else mma<128>(c,a+b*4,xs,b);
 }else mma<NT>(c,a+b*4,xs,b);
}''')
    text = text.replace('int e,int mt,int N,int tile){', 'int e,int mt,int N,int tile,int active){')
    text = text.replace('for(int j=0;j<NT/16;++j){int r=t/8+j*16,k=(t&7)*8;', 'for(int j=0;j<NT/16;++j)if(j*16<active){int r=t/8+j*16,k=(t&7)*8;')
    text = text.replace('template<int NT,int KT,int S,int SPLIT,bool FIRST>', 'template<int NT,int KT,int S,int SPLIT,bool ADAPT,bool FIRST>')
    text = text.replace('int src[NT/16];', 'int remain=min(NT,count-row),active=ADAPT?(remain<=16?16:remain<=32?32:remain<=64?64:128):NT;\n  int src[NT/16];')
    text = text.replace('split*NS+i);', 'split*NS+i,active);').replace('split*NS+k+S-1);', 'split*NS+k+S-1,active);')
    text = text.replace('pair<KT/128', 'pair<(KT>=128?KT/128:1)').replace('sub<2;', 'sub<(KT>=128?2:1);')
    text = text.replace('mma_group<NT,true>(acc,af[reg],bs+st*NT*KT+chunk*NT*64);', 'mma_group<NT,true,ADAPT>(acc,af[reg],bs+st*NT*KT+chunk*NT*64,active);')
    # Select the MMA width outside the K loop. A conditional at every MMA
    # makes ptxas drain pending groups and can spill the fragment array.
    begin = text.index('  float acc[NT/2]={};unsigned af')
    end = text.index('  if constexpr(SPLIT>1)',begin)
    body = text[begin:end]
    body = body.replace('float acc[NT/2]={};unsigned af', 'unsigned af')
    body = body.replace('mma_group<NT,true,ADAPT>', 'mma_group<WIDTH,true,false>')
    header = r'''template<int NT,int WIDTH,int KT,int S,int SPLIT>
__device__ __forceinline__ void core(const int*Q,const unsigned char*SCALE,const __nv_bfloat16*X,
 const int*src,int e,int mt,int N,int split,unsigned char*qs,unsigned char*ss,
 __nv_bfloat16*bs,const unsigned*lut,float*acc){
 constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT;
 constexpr int active=WIDTH;
 int t=threadIdx.x,w=t>>5,l=t&31;
'''
    helper = header+body+'}\n'
    args = '(Q,SCALE,X,src,e,mt,N,split,qs,ss,bs,lut,acc);'
    call = '  float acc[NT/2]={};\n  if constexpr(ADAPT){\n'
    call += '   if(active<=16)core<NT,16,KT,S,SPLIT>'+args+'\n'
    call += '   else if(active<=32)core<NT,32,KT,S,SPLIT>'+args+'\n'
    call += '   else if constexpr(NT>=64){\n'
    call += '    if(active<=64)core<NT,64,KT,S,SPLIT>'+args+'\n'
    call += '    else if constexpr(NT>=128)core<NT,128,KT,S,SPLIT>'+args+'\n   }\n'
    call += '  }else core<NT,NT,KT,S,SPLIT>'+args+'\n'
    text = text[:begin]+call+text[end:]
    marker = 'template<int NT,int KT,int S,int SPLIT,bool ADAPT,bool FIRST>'
    text = text.replace(marker,helper+marker)
    # A row group is the expert for NT128 at M128. Interleave experts before
    # channel tiles so all touched experts supply CTAs at the start of a wave.
    text = text.replace('group=task/MT,mt=task%MT,ix=0;', 'group=task%TASK[32],mt=task/TASK[32],ix=0;')
    # Completion counters are indexed by (group, channel), independent of order.
    text = text.replace('int e=ORDER[ix],row=', 'task=group*MT+mt;\n  int e=ORDER[ix],row=')
    a = text.index('#define FT_CASES(F)')
    b = text.index('\n}', a)
    cases = '#define FT_CASES(F) \\\n ' + ' \\\n '.join(f'F({i},{nt},{kt},{st},{sk},{str(ad).lower()})' for i,(nt,kt,st,sk,ad) in enumerate(CONFIGS))
    text = text[:a] + cases + text[b:]
    text = text.replace('ID,NT,KT,S,SK)', 'ID,NT,KT,S,SK,AD)')
    text = text.replace('fused<NT,KT,S,SK,true>', 'fused<NT,KT,S,SK,AD,true>').replace('fused<NT,KT,S,SK,false>', 'fused<NT,KT,S,SK,AD,false>').replace('fused<NT,KT,S,SK,F>', 'fused<NT,KT,S,SK,AD,F>').replace('fused<NT,KT,S,SK,FIRST>', 'fused<NT,KT,S,SK,AD,FIRST>')
    return text


if __name__ == '__main__':
    print(source())
