"""Local-only arithmetic substitute for Hopper WGMMA; no cuBLAS capture state."""
import triton
import triton.language as tl
@triton.jit
def _mm(X,W,O,T:tl.constexpr,N:tl.constexpr,K:tl.constexpr,XS:tl.constexpr,WS:tl.constexpr):
    r=tl.program_id(0)*32+tl.arange(0,32);c=tl.program_id(1)*32+tl.arange(0,32);kk=tl.arange(0,64)
    acc=tl.zeros((32,32),tl.float32)
    for j in range(triton.cdiv(K,64)):
        a=tl.load(X+r[:,None]*XS+(j*64+kk)[None,:],(r[:,None]<T)&((j*64+kk)[None,:]<K),0)
        b=tl.load(W+c[:,None]*WS+(j*64+kk)[None,:],(c[:,None]<N)&((j*64+kk)[None,:]<K),0)
        acc=tl.dot(a,b.T,acc)
    tl.store(O+r[:,None]*N+c[None,:],acc,(r[:,None]<T)&(c[None,:]<N))
def mm(x,w,out):
    t,k=x.shape;n=w.shape[0]
    _mm[(triton.cdiv(t,32),triton.cdiv(n,32))](x,w,out,t,n,k,x.stride(0),w.stride(0),num_warps=4,num_stages=2)
