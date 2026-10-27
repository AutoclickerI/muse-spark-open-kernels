# I2: heavy residency with full math. Kind 2 pads the heavy ring so one heavy CTA holds an SM and a small
# (94 KiB, 112 regs x 256) or up to three medium CTAs fit beside it; mode 4 + hist are full clones with CTA traces.
ARMS={'BASE':_all(-2,0),'AA':_all(-2,0),'STREAM':_all(7),'H_BASE':_one(0,(-2,4,0,False)),'H_R1':_one(0,(-2,4,2,False)),
 'S_BASE':_one(2,(-2,4,0,False)),'M_BASE':_one(1,(-2,4,0,False)),'MS':[None,(-2,4,0,False),(-2,4,0,False)],
 'R1':[(-2,4,2,False),(-2,4,0,False),(-2,4,0,False)],'BH':[(4,4,1,True),(4,4,1,True),(4,4,1,True)],'R1H':[(4,4,2,True),(4,4,1,True),(4,4,1,True)]}
NAMES=tuple(ARMS)
FULL=tuple(n for n,c in ARMS.items() if all(x is not None and x[0] in (-2,4,12) for x in c))
