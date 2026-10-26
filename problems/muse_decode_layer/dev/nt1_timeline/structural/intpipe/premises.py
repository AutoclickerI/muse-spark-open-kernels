"""Exact format checks and bounded searches; neither is an H200 measurement."""
import itertools,json,struct
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
mag=(0.,.5,1.,1.5,2.,3.,4.,6.)
bf=lambda x:(struct.unpack('<I',struct.pack('<f',x))[0]+0x7fff+((struct.unpack('<I',struct.pack('<f',x))[0]>>16)&1))>>16
count=0
for raw in range(127):
 e,m=raw>>3,raw&7;s=m*2**-9 if e==0 else (1+m/8)*2**(e-7)
 for sign in (-1,1):
  for code in range(16):
   x=mag[code&7]*(-1 if code&8 else 1)*s*sign/256
   half=struct.unpack('<e',struct.pack('<e',x))[0]
   assert half==x and bf(half)==bf(x),(raw,sign,code,x,half)
   count+=1
assert count==4064
# Standard affine magic-nibble decoder is linear in the inserted nibble. It fails E2M1.
linear=[i*.5 for i in range(8)];bad=[i for i in range(8) if linear[i]!=mag[i]]
# Exhaust all embeddings of 3 magnitude bits into one positive BF16 halfword,
# with every disjoint constant base. Permit arbitrary nibble relabeling (sort).
# Affine bias/scale can only change offset and common slope; adjacent-gap ratios must match.
word=np.arange(0x7f80,dtype=np.uint16);matches=[];normal_matches=[];tested=0
for pos in itertools.combinations(range(15),3):
 mask=sum(1<<p for p in pos);base=word[(word&mask)==0];offset=np.array([sum(((q>>i)&1)<<p for i,p in enumerate(pos)) for q in range(8)],dtype=np.uint16)
 bits=base[:,None]|offset[None,:];valid=bits[:,-1]<0x7f80;bits=bits[valid];base=base[valid];vals=(bits.astype(np.uint32)<<16).view(np.float32).astype(np.float64);gaps=np.diff(vals,axis=1);target=np.array([1,1,1,1,2,2,4]);good=np.all(gaps==gaps[:,0:1]*target,axis=1)&(gaps[:,0]>0);tested+=len(base)
 for b in base[good]:
  rec=dict(magnitude_bit_positions=list(pos),base_bits=int(b),nonzero_bias=bool(b));matches.append(rec)
  if b>=128:normal_matches.append(rec)
assert not normal_matches and all(not x['nonzero_bias'] for x in matches),matches
mask=0x81c081c0;rot=lambda k:((mask<<k)|(mask>>(32-k if k else 32)))&0xffffffff
four=[(0,)+p for p in itertools.combinations(range(1,16),3) if all(not(rot(a)&rot(b)) for a,b in itertools.combinations((0,)+p,2))]
assert not four
# A no-shift direct PRMT BF16 extraction needs low-byte bits6,7 and high-byte bits0,7.
# Its source can only use byte-bit positions0,6,7:24 of64 bits in8 packed bytes.
# 16 independently variable4-bit codes require64 independent bits.
report=dict(finite_E2M1_E4M3_products_exact_in_FP16_and_BF16=count,product_definition='FP4*E4M3/256,127 finite magnitude codes x2 scale signs x16 FP4 codes',
 A=dict(single_affine_magic_linear_counterexample=dict(codes=bad,actual=[linear[i] for i in bad],wanted=[mag[i] for i in bad]),
  exhaustive_positive_BF16_three_bit_embeddings=tested,matching_embeddings=matches,nonzero_bias_solutions=normal_matches,
  scope='Uniform affine bias/scale on a bit-permuted three-bit magnitude with a copied sign. Does not exclude multi-piece arithmetic or every possible floating decoder.',
  decision='No exact uniform-bias form found. The existing zero-bias tiny-BF16 multiply is the matching form;all4064-product exactness does not provide a cheaper extraction.'),
 B=dict(four_disjoint_rotations_per_packed_word=four,direct_no_shift_LOP3_PRMT_available_source_bit_positions_per_byte=[0,6,7],available_bits=24,independent_code_bits=64,
  scope='Direct constant-index zero-shift extraction to the existing tiny-BF16 register layout using constant-index byte permutations/bit selections and no widened weight storage;not a lower bound for every decoder.',
  decision='Requested direct one-LOP3/PRMT no-shift form is not a proven implementation;bounded layout family fails. No H200 arm or measured loss.'),
 C=dict(weight_copy='S19 already uses cp.async.bulk with mbarriers;adding TMA is not a new mechanism',arms=['AFFINE preserves B-fill pass/valid control;one64bit base plus immediate loads','PRIVATE moves steady TMA cursor shared fields to owner registers;count residency first']))
(D/'premises.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
