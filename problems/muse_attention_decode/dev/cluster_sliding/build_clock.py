"""Instrument actual preparation/gather/DSM phases without changing the unclocked performance arm."""
import build,build_warp

def source(warp=False,cuda=None):
    s=cuda if cuda is not None else build_warp.source() if warp else build.source()
    a=s.index('DI void rescale(float (&O)')
    helper='''DI void cs_stamp(const VP& p,int slot) {
  if(threadIdx.x==0) {
    unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));
    const int c=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*8;
    ((unsigned long long*)p.part)[c+slot]=t;
  }
}
'''
    s=s[:a]+helper+s[a:]
    s=s.replace('int64_t width,double sc,int64_t stream) {','int64_t width,double sc,int64_t stream,int64_t clockout) {')
    s=s.replace('  VP p={};p.qkv=', '  VP p={};p.part=(float*)clockout;p.qkv=')
    s=s.replace('  asm volatile("griddepcontrol.launch_dependents;"', '  cs_stamp(p,0);\n  asm volatile("griddepcontrol.launch_dependents;"')
    s=s.replace('  asm volatile("griddepcontrol.wait;":::"memory");', '  asm volatile("griddepcontrol.wait;":::"memory");cs_stamp(p,2);')
    s=s.replace('  issue(0,0);issue(1,1);', '  issue(0,0);issue(1,1);cs_stamp(p,1);')
    s=s.replace('  float O[8][4]={}', '  cs_stamp(p,3);\n  float O[8][4]={}')
    s=s.replace('  // All asynchronous use has drained;', '  cs_stamp(p,4);\n  // All asynchronous use has drained;')
    s=s.replace('  __syncthreads();cbar();', '  __syncthreads();cbar();cs_stamp(p,5);')
    s=s.replace('  cbar();\n}', '  cs_stamp(p,6);cbar();cs_stamp(p,7);\n}')
    assert s.count('cs_stamp(p,')>=8
    assert 'stream,int64_t clockout)' in s
    return s
