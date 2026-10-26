"""Measure packet coding on the organizer's actual seed-5353 layer weight tensors."""
import importlib.util,json,sys,hashlib
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[4]
sys.path.insert(0,str(R/'problems/muse_decode_layer'))
spec=importlib.util.spec_from_file_location('official_layer_weights',R/'problems/muse_decode_layer/reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
w=m.generate_weights(5353,device='cuda');rows={}
for role in ('w13','w2'):
 sf=w[role+'_sf'];raw=sf.view(torch.uint8).reshape(32,-1)
 # Raw E4M3 positive codes are monotonically ordered, and are the precision-preserving packet symbols.
 x=raw.reshape(-1,256).long();lo=x.min(1).values;hi=x.max(1).values;fit=hi-lo<32
 # 256 scales/packet: 160B data+1B base; raw packet:256B+1B base-marker. 16 packets/tile.
 # Variable tile offsets are64-bit; compressed/raw tile payloads are aligned to16B.
 ntiles=raw.numel()//4096
 fits=fit.reshape(-1,16);compressed=(fits.sum(1)*160+(16-fits.sum(1))*256+32+15)//16*16
 baseline=ntiles*36864;packed=int((compressed+32768+8).sum())
 rows[role]=dict(scales=raw.numel(),histogram=torch.bincount(raw.flatten().long(),minlength=256).cpu().tolist(),packet256_fit_fraction=float(fit.float().mean()),raw_packets=int((~fit).sum()),tile_count=ntiles,raw_bank_bytes=baseline,coded_bank_bytes_with_8B_tile_offsets=packed,net_saved_bytes=baseline-packed,net_cut_fraction=1-packed/baseline,compressed_tile_bytes_min=int((compressed+32768).min()),compressed_tile_bytes_max=int((compressed+32768).max()))
print(json.dumps(rows,indent=2),flush=True)
(D/'scale_compressibility.json').write_text(json.dumps(dict(seed=5353,source='problems/muse_decode_layer/reference.py generate_weights; actual complete W13/W2 tensors',source_sha256=hashlib.sha256((R/'problems/muse_decode_layer/reference.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
