
import numpy as np
from scipy.linalg import expm
from oil.utils.utils import Named,export
import jax
import jax.numpy as jnp
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from core.linear_operator_jax import LinearOperator
from jax import jit,vmap
from functools import partial
import logging
from core.product_sum_reps import lazy_kron,lazy_kronsum,LazyPerm,I

def rel_err(A,B):
    return jnp.mean(jnp.abs(A-B))/(jnp.mean(jnp.abs(A)) + jnp.mean(jnp.abs(B))+1e-6)

class Group(object,metaclass=Named):
    lie_algebra = NotImplemented
    #lie_algebra_lazy = NotImplemented
    discrete_generators = NotImplemented
    #discrete_generators_lazy = NotImplemented
    z_scale=None # For scale noise for sampling elements
    is_orthogonal=None
    is_regular = None
    def __init__(self,*args,**kwargs):
        # # Set dense lie_algebra using lie_algebra_lazy if applicable
        # if self.lie_algebra is NotImplemented and self.lie_algebra_lazy is not NotImplemented:
        #     Idense = np.eye(self.lie_algebra_lazy[0].shape[0])
        #     self.lie_algebra = np.stack([h@Idense for h in self.lie_algebra_lazy])
        # # Set dense discrete_generators using discrete_generators_lazy if applicable
        # if self.discrete_generators is NotImplemented and self.discrete_generators_lazy is not NotImplemented:
        #     Idense = np.eye(self.discrete_generators_lazy[0].shape[0])
        #     self.discrete_generators = np.stack([h@Idense for h in self.discrete_generators_lazy])

        if self.lie_algebra is NotImplemented:
            self.lie_algebra = np.zeros((0,self.d,self.d))
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = np.zeros((0,self.d,self.d))
    
        self.args = args
        if isinstance(self.lie_algebra,np.ndarray): self.lie_algebra = jax.device_put(self.lie_algebra)
        if isinstance(self.discrete_generators,np.ndarray): self.discrete_generators = jax.device_put(self.discrete_generators)

        # Set orthogonal flag automatically if not specified
        if self.is_regular: self.is_orthogonal=True
        if self.is_orthogonal is None:
            self.is_orthogonal = True
            if len(self.lie_algebra)!=0:
                A_dense =jnp.stack([Ai@jnp.eye(self.d) for Ai in self.lie_algebra])
                self.is_orthogonal &= rel_err(-A_dense.transpose((0,2,1)),A_dense)<1e-6
            if len(self.discrete_generators)!=0:
                h_dense = jnp.stack([hi@jnp.eye(self.d) for hi in self.discrete_generators])
                self.is_orthogonal &= rel_err(h_dense.transpose((0,2,1))@h_dense,jnp.eye(self.d))<1e-6

        # Set regular flag automatically if not specified
        if self.is_orthogonal and (self.is_regular is None):
            self.is_regular=True
            self.is_regular &= (len(self.lie_algebra)==0) # no infinitesmal generators and all rows have one 1
            if len(self.discrete_generators)!=0:
                h_dense = jnp.stack([hi@jnp.eye(self.d) for hi in self.discrete_generators])
                self.is_regular &= ((h_dense==1).astype(np.int).sum(-1)==1).all()
        

    def exp(self,A):
        return expm(A)
    def num_constraints(self):
        return len(self.lie_algebra)+len(self.discrete_generators)
    @property
    def d(self):
        if self.lie_algebra is not NotImplemented and len(self.lie_algebra):
            return self.lie_algebra[0].shape[-1]
        if self.discrete_generators is not NotImplemented and len(self.discrete_generators):
            return self.discrete_generators[0].shape[-1]
        return self._d

    def sample(self):
        return self.samples(1)[0]

    def samples(self,N):
        A_dense = jnp.stack([Ai@jnp.eye(self.d) for Ai in self.lie_algebra]) if len(self.lie_algebra) else jnp.zeros((0,self.d,self.d))
        h_dense = jnp.stack([hi@jnp.eye(self.d) for hi in self.discrete_generators]) if len(self.discrete_generators) else jnp.zeros((0,self.d,self.d))
        z = np.random.randn(N,A_dense.shape[0])
        if self.z_scale is not None:
            z*= self.z_scale
        k = np.random.randint(-5,5,size=(N,h_dense.shape[0],3))
        jax_seed=  np.random.randint(100)
        return noise2samples(z,k,A_dense,h_dense,jax_seed)

    def check_valid_group_elems(self,g):
        return True

    def __and__(self,G2):
        return CombinedGenerators(self,G2)
    def __str__(self):
        return repr(self)
    def __repr__(self):
        outstr = f"{self.__class__}"
        if self.args:
            outstr += '('+''.join(repr(arg) for arg in self.args)+')'
        return outstr
    def __eq__(self,G2): #TODO: check that spans are equal?
        return repr(self)==repr(G2)
        # if self.lie_algebra.shape!=G2.lie_algebra.shape or \
        #     self.discrete_generators.shape!=G2.discrete_generators.shape:
        #     return False
        # return  (self.lie_algebra==G2.lie_algebra).all() and (self.discrete_generators==G2.discrete_generators).all()
    def __hash__(self):
        return hash(repr(self))
        # algebra = jax.device_get(self.lie_algebra).tobytes()
        # gens = jax.device_get(self.discrete_generators).tobytes()
        # return hash((algebra,gens,self.lie_algebra.shape,self.discrete_generators.shape))
    def __lt__(self, other):
        return hash(self) < hash(other) #For sorting purposes only

    def __mul__(self,other):
        return DirectProduct(self,other)

@jit
def matrix_power_simple(M,n):
    out = jnp.eye(M.shape[-1])
    body = lambda Mn: jax.lax.fori_loop(0,Mn[1],lambda i,g: Mn[0]@g,out)
    out = jax.lax.cond(n<0,(jnp.linalg.inv(M),-n),body,(M,n),body)
    return out
    
@jit
def noise2sample(z,ks,lie_algebra,discrete_generators,seed=0):
    """ [zs (D,)] [ks (M,K)] [lie_algebra (D,d,d)] [discrete_generators (M,d,d)] 
        Here K is the number of repeats for a given discrete generator."""
    g = jnp.eye(lie_algebra.shape[-1])
    if lie_algebra.shape[0]:
        A = (z[:,None,None]*lie_algebra).sum(0)
        g = g@jax.scipy.linalg.expm(A)
    key = jax.random.PRNGKey(seed)
    M,K = ks.shape
    if M==0: return g
    for k in range(K): # multiple rounds of discrete generators
        key,pkey = jax.random.split(key)
        for i in jax.random.permutation(pkey,M): # Randomize the order of generators
            g = g@matrix_power_simple(discrete_generators[i],ks[i,k])#jnp.linalg.matrix_power(discrete_generators[i],ks[i])
    return g

@jit
def noise2samples(zs,ks,lie_algebra,discrete_generators,seed=0):
    return vmap(noise2sample,(0,0,None,None,None),0)(zs,ks,lie_algebra,discrete_generators,seed)




class DirectProduct(Group):
    def __init__(self,G1,G2):
        I1,I2 = I(G1.d),I(G2.d)
        self.lie_algebra = [lazy_kronsum([A1,0*I2]) for A1 in G1.lie_algebra]+[lazy_kronsum([0*I1,A2]) for A2 in G2.lie_algebra]
        self.discrete_generators = [lazy_kron([M1,I2]) for M1 in G1.discrete_generators]+[lazy_kron([I1,M2]) for M2 in G2.discrete_generators]
        self.names = (repr(G1),repr(G2))
        super().__init__()
        
    def __repr__(self):
        return f"{self.names[0]}x{self.names[1]}"

class WreathProduct(Group):
    def __init__(self,G1,G2):
        raise NotImplementedError

class SemiDirectProduct(Group):
    def __init__(self,G1,G2):
        raise NotImplementedError

@export
class Trivial(Group): #""" The trivial group G={I} in N dimensions """
    def __init__(self,N):
        self._d = N
        super().__init__(N)

@export
class SO(Group): #""" The special orthogonal group SO(N) in N dimensions"""
    def __init__(self,N):
        self.lie_algebra = np.zeros(((N*(N-1))//2,N,N))
        k=0
        for i in range(N):
            for j in range(i):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,j,i] = -1
                k+=1
        super().__init__(N)
@export
class O(SO): #""" The Orthogonal group O(N) in N dimensions"""
    def __init__(self,N):
        self.discrete_generators = np.eye(N)[None]
        self.discrete_generators[0,0,0]=-1
        super().__init__(N)
@export
class C(Group): #""" The Cyclic group Ck in 2 dimensions"""
    def __init__(self,k):
        theta = 2*np.pi/k
        self.discrete_generators = np.zeros((1,2,2))
        self.discrete_generators[0,:,:] = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        super().__init__(k)
@export
class D(C): #""" The Dihedral group Dk in 2 dimensions"""
    def __init__(self,k):
        super().__init__(k)
        self.discrete_generators = np.concatenate((self.discrete_generators,np.array([[[-1,0],[0,1]]])))
@export
class Scaling(Group):
    def __init__(self,N):
        self.lie_algebra = np.eye(N)[None]
        super().__init__(N)

class Parity(Group): #""" The spacial parity group in 1+3 dimensions"""
    discrete_generators = -np.eye(4)[None]
    discrete_generators[0,0,0] = 1

class TimeReversal(Group): #""" The time reversal group in 1+3 dimensions"""
    discrete_generators = np.eye(4)[None]
    discrete_generators[0,0,0] = -1

@export
class SO13p(Group): #""" The component of Lorentz group connected to identity"""
    lie_algebra = np.zeros((6,4,4))
    lie_algebra[3:,1:,1:] = SO(3).lie_algebra
    for i in range(3):
        lie_algebra[i,1+i,0] = lie_algebra[i,0,1+i] = 1.

    # Adjust variance for samples along boost generators. For equivariance checks
    # the exps for high order tensors can get very large numbers
    z_scale = np.array([.3,.3,.3,1,1,1])
@export
class SO13(SO13p):
    discrete_generators = -np.eye(4)[None]

@export
class O13(SO13p):
    discrete_generators = np.eye(4)[None] +np.zeros((2,1,1))
    discrete_generators[0] *= -1
    discrete_generators[1,0,0] = -1
@export
class Lorentz(O13): pass

@export
class Sp(Group):
    def __init__(self,m):
        self.lie_algebra = np.zeros((m*(2*m+1),2*m,2*m))
        k=0
        for i in range(m): # block diagonal elements
            for j in range(m):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,m+j,m+i] = -1
                k+=1
        for i in range(m):
            for j in range(i+1):
                self.lie_algebra[k,m+i,j] = 1
                self.lie_algebra[k,m+j,i] = 1
                k+=1
                self.lie_algebra[k,i,m+j] = 1
                self.lie_algebra[k,j,m+i] = 1
                k+=1
        super().__init__(m)

@export
class Symplectic(Sp): pass

class LazyShift(LinearOperator):
    def __init__(self,n,k=1):
        self.k=k
        self.shape = (n,n)

    def _matmat(self,V): #(c,k) #Still needs to be tested??
        return jnp.roll(V,self.k,axis=0)
    def _matvec(self,V):
        return jnp.roll(V,self.k,axis=0)
    def _adjoint(self):
        return LazyShift(self.shape[0],-self.k)
    def invT(self):
        return self

@export
class Z(Group):
    def __init__(self,n):
        self.discrete_generators = [LazyShift(n)]
        super().__init__(n)

@export
class DiscreteTranslation(Z): pass # Alias cyclic translations with Z

@export
class S(Group): #The permutation group
    def __init__(self,n):
        #K=n//5
        # perms = np.arange(n)[None]+np.zeros((K,1)).astype(int)
        # for i in range(1,K):
        #     perms[i,[0,(i*n)//K]] = perms[i,[(i*n)//K,0]]
        # print(perms)
        # self.discrete_generators = [LazyPerm(perm) for perm in perms]+[LazyShift(n)]
        perms = np.arange(n)[None]+np.zeros((n-1,1)).astype(int)
        perms[:,0] = np.arange(1,n)
        perms[np.arange(n-1),np.arange(1,n)[None]]=0
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        #self.discrete_generators = [SwapMatrix((0,i),n) for i in range(1,n)]
        # Adding superflous extra generators can actually *decrease* the runtime of the iterative
        # krylov solver by improving the conditioning of the constraint matrix
        # swap_perm = np.arange(n).astype(int)
        # swap_perm[[0,1]] = swap_perm[[1,0]]
        # swap_perm2 = np.arange(n).astype(int)
        # swap_perm2[[0,n//2]] = swap_perm2[[n//2,0]]
        # self.discrete_generators = [LazyPerm(swap_perm)]+[LazyShift(n,2**i) for i in range(int(np.log2(n)))]
        super().__init__(n)

@export
class Permutation(S): pass #Alias permutation group with Sn.

@export
class U(Group): # Of dimension n^2
    def __init__(self,n):
        lie_algebra_real = np.zeros((n**2,n,n))
        lie_algebra_imag = np.zeros((n**2,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k,i,j] = 1
                lie_algebra_real[k,j,i] = -1
                k+=1
                # symmetric imaginary generators
                lie_algebra_imag[k,i,j] = 1
                lie_algebra_imag[k,j,i] = 1
                k+=1
        for i in range(n):
            # diagonal imaginary generators
            lie_algebra_imag[k,i,i] = 1
            k+=1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag*1j
        super().__init__(n)
@export
class SU(Group): # Of dimension n^2-1
    def __init__(self,n):
        if n==1: return Trivial(1)
        lie_algebra_real = np.zeros((n**2-1,n,n))
        lie_algebra_imag = np.zeros((n**2-1,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k,i,j] = 1
                lie_algebra_real[k,j,i] = -1
                k+=1
                # symmetric imaginary generators
                lie_algebra_imag[k,i,j] = 1
                lie_algebra_imag[k,j,i] = 1
                k+=1
        for i in range(n-1):
            # diagonal traceless imaginary generators
            lie_algebra_imag[k,i,i] = 1
            for j in range(n):
                if i==j: continue
                lie_algebra_imag[k,j,j] = -1/(n-1)
            k+=1
        self.lie_algebra = lie_algebra_real + lie_algebra_imag*1j
        super().__init__(n)

@export
class Cube(Group):
    # A discrete version of SO(3) including all 90 degree rotations in 3d space
    # Implements a 6 dimensional representation on the faces of a cube
    def __init__(self):
        order = np.arange(6) # []
        Fperm = np.array([4,1,0,3,5,2])
        Lperm = np.array([3,0,2,5,4,1])
        self.discrete_generators = [LazyPerm(perm) for perm in [Fperm,Lperm]]
        super().__init__()


def pad(permutation):
    assert len(permutation)==48
    padded = np.zeros((6,9)).astype(permutation.dtype)
    padded[:,:4] = permutation.reshape(6,8)[:,:4]
    padded[:,5:] = permutation.reshape(6,8)[:,4:]
    return padded

def unpad(padded_perm):
    return np.concatenate([padded_perm[:,:4],padded_perm[:,5:]],-1).reshape(-1)




class SwapMatrix(LinearOperator):
    def __init__(self,swaprows,n):
        self.swaprows=swaprows
        self.shape = (n,n)
    def _matmat(self,V): #(c,k)
        V = jax.ops.index_update(V, jax.ops.index[self.swaprows], V[self.swaprows[::-1]])
        return V
    def _matvec(self,V):
        return self._matmat(V)
    def _adjoint(self):
        return self
    def invT(self):
        return self

@export
class RubiksCube(Group): #3x3 rubiks cube
    def __init__(self):
        #Faces are ordered U,F,R,B,L,D (the net of the cube) #    B
        order = np.arange(48)                                #  L U R
        order_padded = pad(order) #include a center element  #    F
        # Compute permutation for Up quarter turn            #    D
        order_padded[0,:] = np.rot90(order_padded[0].reshape(3,3),1).reshape(9) # Rotate top face
        FRBL = np.array([1,2,3,4])
        order_padded[FRBL,:3] = order_padded[np.roll(FRBL,1),:3] # F <- L,R <- F,B <- R,L <- B
        Uperm = unpad(order_padded)
        # Now form all other generators by using full rotations of the cube by 90 clockwise about a given face
        RotFront =pad(np.arange(48))# rotate full cube so that Left face becomes Up, Up becomes Right, Right becomes Down, Down becomes Left
        URDL = np.array([0,2,5,4])
        RotFront[URDL,:] = RotFront[np.roll(URDL,1),:]
        RotFront = unpad(RotFront)
        RotBack = np.argsort(RotFront)
        RotLeft = pad(np.arange(48))
        UFDB = np.array([0,1,5,3])
        RotLeft[UFDB,:] = RotLeft[np.roll(UFDB,1),:]
        RotLeft = unpad(RotLeft)
        RotRight = np.argsort(RotLeft)

        Fperm = RotRight[Uperm[RotLeft]] # Fperm = RotLeft<-Uperm<-RotRight
        Rperm = RotBack[Uperm[RotFront]] # Rperm = RotFront<-Uperm<-RotBack
        Bperm = RotLeft[Uperm[RotRight]]# Bperm = RotRight<-Uperm<-RotLeft
        Lperm = RotFront[Uperm[RotBack]] # Lperm = RotBack<-Uperm<-RotFront
        Dperm = RotRight[RotRight[Uperm[RotLeft[RotLeft]]]] # Dperm = RotLeft<-RotLeft<-Uperm<-RotRight<-RotRight
        self.discrete_generators = [LazyPerm(perm) for perm in [Uperm,Fperm,Rperm,Bperm,Lperm,Dperm]]
        super().__init__()


@export
class RubiksCube2x2(Group):
    def __init__(self):
        #Faces are ordered U,F,R,B,L,D (the net of the cube) #    B
        Uperm = np.arange(24).reshape(6,4)                   #  L U R
                                  #include a center element  #    F
        # Compute permutation for Up quarter turn            #    D
        Uperm[0,:] = np.rot90(Uperm[0].reshape(2,2),-1).reshape(4) # Rotate top face clockwise
        FRBL = np.array([1,2,3,4])
        Uperm[FRBL,:2] = Uperm[np.roll(FRBL,-1),:2] # F <- L,R <- F,B <- R,L <- B, but only 1st 2 elems
        Uperm = Uperm.reshape(-1)
        # Now form all other generators by using full rotations of the cube by 90 clockwise about a given face
        RotFront =np.arange(24).reshape(6,4)# rotate full cube so that Left face becomes Up, Up becomes Right, Right becomes Down, Down becomes Left
        URDL = np.array([0,2,5,4])
        RotFront[URDL,:] = RotFront[np.roll(URDL,1),:] #clockwise about F
        RotFront = RotFront.reshape(-1)
        RotBack = np.argsort(RotFront)
        RotLeft = np.arange(24).reshape(6,4)
        UFDB = np.array([0,1,5,3])
        RotLeft[UFDB,:] = RotLeft[np.roll(UFDB,1),:]
        RotLeft = RotLeft.reshape(-1)
        RotRight = np.argsort(RotLeft)

        Fperm = RotRight[Uperm[RotLeft]] # Fperm = RotLeft<-Uperm<-RotRight
        Rperm = RotBack[Uperm[RotFront]] # Rperm = RotFront<-Uperm<-RotBack
        Bperm = RotLeft[Uperm[RotRight]]# Bperm = RotRight<-Uperm<-RotLeft
        Lperm = RotFront[Uperm[RotBack]] # Lperm = RotBack<-Uperm<-RotFront
        Dperm = RotRight[RotRight[Uperm[RotLeft[RotLeft]]]] # Dperm = RotLeft<-RotLeft<-Uperm<-RotRight<-RotRight
        I = np.eye(24)
        self.perms = [Uperm,Fperm,Rperm,Bperm,Lperm,Dperm]
        self.discrete_generators = [LazyPerm(perm) for perm in [Uperm,Fperm,Rperm,Bperm,Lperm,Dperm]]
        super().__init__()



class Rot90(LinearOperator):
    def __init__(self,n,k):
        self.shape = (n*n,n*n)
        self.n=n
        self.k = k
    def _matmat(self,V): #(c,k)
        return jnp.rot90(V.reshape((self.n,self.n,-1)),self.k).reshape(V.shape)
    def _matvec(self,V):
        return jnp.rot90(V.reshape((self.n,self.n,-1)),self.k).reshape(V.shape)
    def invT(self):
        return self


@export
class ZnxZn(Group):
    def __init__(self,n):
        Zn = Z(n)
        nshift = Zn.discrete_generators[0]
        In = I(n)
        Idense = np.eye(n*n)
        self.discrete_generators = [lazy_kron([nshift,In]),lazy_kron([In,nshift])]
        super().__init__(n)

@export
class ZksZnxZn(Group):
    def __init__(self,k,n):
        Zn = Z(n)
        Zk = Z(k)
        nshift = Zn.discrete_generators[0]
        kshift = Zk.discrete_generators[0]
        In = I(n)
        Ik = I(k)
        Idense = np.eye(k*n*n)
        assert k in [2,4]
        self.discrete_generators = [lazy_kron([Ik,nshift,In]),lazy_kron([Ik,In,nshift]),lazy_kron([kshift,Rot90(n,4//k)])]
        super().__init__(k,n)

class Embed(Group):
    def __init__(self,G,d,slice):
        self.lie_algebra = np.zeros((G.lie_algebra.shape[0],d,d))
        self.discrete_generators = np.zeros((G.discrete_generators.shape[0],d,d))
        self.discrete_generators += np.eye(d)
        self.lie_algebra[:,slice,slice] = G.lie_algebra
        self.discrete_generators[:,slice,slice]  =G.discrete_generators
        self.name = f"{G}_R{d}"
        super().__init__()
        
    def __repr__(self):
        return self.name

@export
def SO2eR3():
    return Embed(SO(2),3,slice(2))

@export
def O2eR3():
    return Embed(O(2),3,slice(2))

@export
def DkeR3(k):
    return Embed(D(k),3,slice(2))