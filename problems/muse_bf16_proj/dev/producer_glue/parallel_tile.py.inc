@triton.jit
def _ps_tile(P,A,B,STATS,STREAM,ALPHA,BETA,POSTW,pn,
             T:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BN:tl.constexpr,TB:tl.constexpr,
             ROLE:tl.constexpr,POST:tl.constexpr):
    # GEMV needs TB>=16 for MMA; its epilogue does not. Keeping that padding
    # makes M=1 statistics perform reductions over fifteen unused token lanes.
    TV:tl.constexpr=1 if T<=1 else 2 if T<=2 else 4 if T<=4 else 8 if T<=8 else TB
    t=tl.arange(0,TV)
    if ROLE==1:
        j=pn*BN+tl.arange(0,BN)
        br=tl.full((BN,TV),0.,tl.float32)
        for s in tl.static_range(S):
            br+=tl.load(P+s*T*N+t[None,:]*N+j[:,None],t[None,:]<T,0.,cache_modifier='.cg')
        v=tl.where(j[:,None]<4096,br,0.)
        tl.store(B+t[None,:]*N+j[:,None],br,t[None,:]<T)
        tl.store(STATS+(t*(N//BN)+pn)*4,tl.sum(v*v,0),t<T)
    else:
        # A sliced branch has eight residual slices. Issue their independent
        # loads together, instead of consuming one slice before loading the
        # next. At T>4 use two groups to bound the register working set.
        V:tl.constexpr=1 if N==8192 else 8 if T<=4 else 4
        i=tl.arange(0,BN*V)
        j=pn*BN+i%BN
        m0=tl.full((TV,),0.,tl.float32);m1=m0;m2=m0;m3=m0
        for group in tl.static_range(8192//N//V):
            c=j+(i//BN+group*V)*N
            z=tl.full((BN*V,TV),0.,tl.float32)
            for s in tl.static_range(S):
                z+=tl.load(P+s*T*N+t[None,:]*N+j[:,None],(t[None,:]<T)&(c[:,None]<N),0.,cache_modifier='.cg')
            if POST:z*=tl.load(POSTW+c)[:,None]
            u=tl.load(ALPHA+c)[:,None]*tl.load(STREAM+t[None,:]*8192+c[:,None],t[None,:]<T,0.)
            v=tl.load(BETA+c)[:,None]*z
            m0+=tl.sum(z*z,0);m1+=tl.sum(u*u,0)
            m2+=tl.sum(u*v,0);m3+=tl.sum(v*v,0)
            tl.store(A+t[None,:]*8192+c[:,None],u,t[None,:]<T)
            tl.store(B+t[None,:]*8192+c[:,None],v,t[None,:]<T)
        dst=STATS+(t*(N//BN)+pn)*4
        tl.store(dst,m0,t<T);tl.store(dst+1,m1,t<T)
        tl.store(dst+2,m2,t<T);tl.store(dst+3,m3,t<T)
