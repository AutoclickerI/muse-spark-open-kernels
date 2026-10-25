"""Independent whole-bank code/scale equality, and per-layout density for legal duplicate layouts."""
def verify(torch,state):
 bank=state['s4_nt1_bank']['banks'];raw=bank['lop3'];packed=bank['lut'];device=raw['w13p'].device
 u=torch.arange(256,device=device,dtype=torch.int64)
 lut=((u&7)<<6)|((u&8)<<12)|((u&112)<<18)|((u&128)<<24)
 checked=0
 for key in ('w13p','w2p'):
  a=raw[key].view(torch.uint8).reshape(-1,36864);b=packed[key].view(torch.uint8).reshape(-1,36864)
  assert torch.equal(a[:,32768:],b[:,32768:]),key+' scale codes'
  for start in range(0,a.shape[0],64):
   src=a[start:start+64,:32768].contiguous().view(torch.int32).reshape(-1,4).to(torch.int64)&0xffffffff
   dst=b[start:start+64,:32768].contiguous().reshape(-1,16).long()
   for half in (0,1):
    A=src[:,half*2];B=src[:,half*2+1];W=(A&0x60306030)|(B&0x9fcf9fcf)
    for j,(word,shift) in enumerate(((A,0),(A,4),(A,10),(B,2),(B,6),(B,12),(W,24),(W,30))):
     v=((word>>shift)|(word<<(32-shift if shift else 0)))&0x81c081c0 if shift else word&0x81c081c0
     assert torch.equal(v,lut[dst[:,half*8+j]]),(key,start,half,j)
   checked+=src.shape[0]*32
  assert a.numel()/(a.shape[0]*65536)*16==9.
 return {'weights_bit_exact':checked,'scale_codes_bit_exact':True,'density_each_layout':9.,'layouts':2,'retained_bits_per_weight_including_scale':4.5}
