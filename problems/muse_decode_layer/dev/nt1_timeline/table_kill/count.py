"""Count SASS between the two CS2R clock reads of each kernel; report instructions per 16 weights (256 weights per lane)."""
import re,subprocess,sys
sass=subprocess.run(['cuobjdump','-sass',sys.argv[1]],capture_output=True,text=True).stdout
for name,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
    ins=[l.split(';')[0].split('*/',1)[-1].strip() for l in body.splitlines() if re.match(r'\s*/\*[0-9a-f]{4}\*/',l)]
    marks=[i for i,l in enumerate(ins) if 'CS2R' in l and 'SR_CLOCK' in l]
    if len(marks)<2:print(name,'no markers',len(ins));continue
    loop=ins[marks[0]+1:marks[-1]];mma=sum('HMMA' in l for l in loop)
    ops={};[ops.__setitem__(l.split()[0].split('.')[0],ops.get(l.split()[0].split('.')[0],0)+1) for l in loop if l]
    print(f'{name}: {len(loop)} SASS for 256 weights/lane = {len(loop)/16:.2f} per 16 weights; non-MMA {(len(loop)-mma)/16:.2f}; HMMA {mma}; mix {dict(sorted(ops.items(),key=lambda x:-x[1]))}')
