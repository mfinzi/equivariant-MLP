
import numpy as np
from scipy.linalg import expm
from emlp.utils import Named,export
import jax
import jax.numpy as jnp
from emlp.reps.linear_operators import LazyShift,SwapMatrix,Rot90,LazyKron,LazyKronsum,LazyPerm,I
from jax import jit,vmap

def rel_err(A,B):
    return jnp.mean(jnp.abs(A-B))/(jnp.mean(jnp.abs(A)) + jnp.mean(jnp.abs(B))+1e-6)

@export
class Group(object,metaclass=Named):
    """ Abstract Group Object which new groups should inherit from. """
    lie_algebra = NotImplemented  #: The continuous generators
    discrete_generators = NotImplemented  #: The discrete generators
    z_scale=None # For scale noise for sampling elements
    is_orthogonal=None
    is_permutation = None
    d = NotImplemented  #: The dimension of the base representation
    def __init__(self,*args,**kwargs):
        # get the dimension of the base group representation
        if self.d is NotImplemented: 
            if self.lie_algebra is not NotImplemented and len(self.lie_algebra):
                self.d= self.lie_algebra[0].shape[-1]
            if self.discrete_generators is not NotImplemented and len(self.discrete_generators):
                self.d= self.discrete_generators[0].shape[-1]

        if self.lie_algebra is NotImplemented:
            self.lie_algebra = np.zeros((0,self.d,self.d))
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = np.zeros((0,self.d,self.d))

        self.args = args
        if isinstance(self.lie_algebra,np.ndarray): self.lie_algebra = jax.device_put(self.lie_algebra)
        if isinstance(self.discrete_generators,np.ndarray): self.discrete_generators = jax.device_put(self.discrete_generators)

        # Set orthogonal flag automatically if not specified
        if self.is_permutation: self.is_orthogonal=True
        if self.is_orthogonal is None:
            self.is_orthogonal = True
            if len(self.lie_algebra)!=0:
                A_dense =jnp.stack([Ai@jnp.eye(self.d) for Ai in self.lie_algebra])
                self.is_orthogonal &= rel_err(-A_dense.transpose((0,2,1)),A_dense)<1e-6
            if len(self.discrete_generators)!=0:
                h_dense = jnp.stack([hi@jnp.eye(self.d) for hi in self.discrete_generators])
                self.is_orthogonal &= rel_err(h_dense.transpose((0,2,1))@h_dense,jnp.eye(self.d))<1e-6

        # Set regular flag automatically if not specified
        if self.is_orthogonal and (self.is_permutation is None):
            self.is_permutation=True
            self.is_permutation &= (len(self.lie_algebra)==0)  # no infinitesmal generators and all rows have one 1
            if len(self.discrete_generators)!=0:
                h_dense = jnp.stack([hi@jnp.eye(self.d) for hi in self.discrete_generators])
                self.is_permutation &= ((h_dense==1).astype(int).sum(-1)==1).all()

    def exp(self,A):
        """ Matrix exponential """
        return expm(A)

    def num_constraints(self):
        return len(self.lie_algebra)+len(self.discrete_generators)

    def sample(self):
        """Draw a sample from the group (not necessarily Haar measure)"""
        return self.samples(1)[0]

    def samples(self,N):
        """ Draw N samples from the group (not necessarily Haar measure)"""
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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        outstr = f"{self.__class__}"
        if self.args:
            outstr += '('+''.join(repr(arg) for arg in self.args)+')'
        return outstr

    def __eq__(self,G2):  # TODO: more permissive by checking that spans are equal?
        return repr(self)==repr(G2)
        
    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        """ For sorting purposes only """
        return hash(self) < hash(other)

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

@export
class Trivial(Group):
    """ The trivial group G={I} in n dimensions. If you want to see how the
        inductive biases of EMLP perform without any symmetry, use Trivial(n)"""
    def __init__(self,n):
        self.d = n
        super().__init__(n)

@export
class SO(Group):
    """ The special orthogonal group SO(n) in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = np.zeros(((n*(n-1))//2,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,j,i] = -1
                k+=1
        super().__init__(n)

@export
class O(SO):
    """ The Orthogonal group O(n) in n dimensions"""
    def __init__(self,n):
        self.discrete_generators = np.eye(n)[None]
        self.discrete_generators[0,0,0]=-1
        super().__init__(n)

@export
class C(Group):
    """ The Cyclic group Ck in 2 dimensions"""
    def __init__(self,k):
        theta = 2*np.pi/k
        self.discrete_generators = np.zeros((1,2,2))
        self.discrete_generators[0,:,:] = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        super().__init__(k)
@export
class D(C):
    """ The Dihedral group Dk in 2 dimensions"""
    def __init__(self,k):
        super().__init__(k)
        self.discrete_generators = np.concatenate((self.discrete_generators,np.array([[[-1,0],[0,1]]])))
@export
class Scaling(Group):
    """ The scaling group in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = np.eye(n)[None]
        super().__init__(n)

class Parity(Group):  # """ The spacial parity group in 1+3 dimensions"""
    discrete_generators = -np.eye(4)[None]
    discrete_generators[0,0,0] = 1

class TimeReversal(Group):  # """ The time reversal group in 1+3 dimensions"""
    discrete_generators = np.eye(4)[None]
    discrete_generators[0,0,0] = -1

@export
class SO13p(Group):
    """ The component of Lorentz group connected to identity"""
    lie_algebra = np.zeros((6,4,4))
    lie_algebra[3:,1:,1:] = SO(3).lie_algebra
    for i in range(3):
        lie_algebra[i,1+i,0] = lie_algebra[i,0,1+i] = 1.

    # Adjust variance for samples along boost generators. For equivariance checks
    # the exps for high order tensors can get very large numbers
    z_scale = np.array([.3,.3,.3,1,1,1]) # can get rid of now

@export
class SO13(SO13p):
    discrete_generators = -np.eye(4)[None]

@export
class O13(SO13p):
    """ The full lorentz group (including Parity and Time reversal)"""
    discrete_generators = np.eye(4)[None] +np.zeros((2,1,1))
    discrete_generators[0] *= -1
    discrete_generators[1,0,0] = -1
@export
class Lorentz(O13): pass

@export
class SO11p(Group):
    """ The identity component of O(1,1) (Lorentz group in 1+1 dimensions)"""
    lie_algebra = np.array([[0.,1.],[1.,0.]])[None]

@export
class O11(SO11p):
    """ The Lorentz group O(1,1) in 1+1 dimensions """
    discrete_generators = np.eye(2)[None]+np.zeros((2,1,1))
    discrete_generators[0]*=-1
    discrete_generators[1,0,0] = -1

@export
class Sp(Group):
    """ Symplectic group Sp(m) in 2m dimensions (sometimes referred to
        instead as Sp(2m) )"""
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
class Z(Group):
    r""" The cyclic group Z_n (discrete translation group) of order n.
        Features a regular base representation."""
    def __init__(self,n):
        self.discrete_generators = [LazyShift(n)]
        super().__init__(n)

@export
class S(Group): #The permutation group
    r""" The permutation group S_n with an n dimensional regular representation."""
    def __init__(self,n):
        # Here we choose n-1 generators consisting of swaps between the first element
        # and every other element
        perms = np.arange(n)[None]+np.zeros((n-1,1)).astype(int)
        perms[:,0] = np.arange(1,n)
        perms[np.arange(n-1),np.arange(1,n)[None]]=0
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        super().__init__(n)
        # We can also have chosen the 2 generator soln described in the paper, but
        # adding superflous extra generators surprisingly can sometimes actually *decrease* 
        # the runtime of the iterative krylov solver by improving the conditioning 
        # of the constraint matrix

@export
class SL(Group):
    """ The special linear group SL(n) in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = np.zeros((n*n-1,n,n))
        k=0
        for i in range(n):
            for j in range(n):
                if i==j: continue #handle diag elements separately
                self.lie_algebra[k,i,j] = 1
                k+=1
        for l in range(n-1):
            self.lie_algebra[k,l,l] = 1
            self.lie_algebra[k,-1,-1] = -1
            k+=1
        super().__init__(n)

@export
class GL(Group):
    """ The general linear group GL(n) in n dimensions"""
    def __init__(self,n):
        self.lie_algebra = np.zeros((n*n,n,n))
        k=0
        for i in range(n):
            for j in range(n):
                self.lie_algebra[k,i,j] = 1
                k+=1
        super().__init__(n)

@export
class U(Group):  # Of dimension n^2
    """ The unitary group U(n) in n dimensions (complex)"""
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
class SU(Group):  # Of dimension n^2-1
    """ The special unitary group SU(n) in n dimensions (complex)"""
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
    """ A discrete version of SO(3) including all 90 degree rotations in 3d space
    Implements a 6 dimensional representation on the faces of a cube"""
    def __init__(self):
        #order = np.arange(6) # []
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




@export
class RubiksCube(Group): #3x3 rubiks cube
    r""" The Rubiks cube group G<S_48 consisting of all valid 3x3 Rubik's cube transformations.
        Generated by the a quarter turn about each of the faces."""
    def __init__(self):
        # Faces are ordered U,F,R,B,L,D (the net of the cube) #    B
        order = np.arange(48)                                 #  L U R
        order_padded = pad(order) # include a center element  #    F
        # Compute permutation for Up quarter turn             #    D
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

        Fperm = RotRight[Uperm[RotLeft]]  # Fperm = RotLeft<-Uperm<-RotRight
        Rperm = RotBack[Uperm[RotFront]]  # Rperm = RotFront<-Uperm<-RotBack
        Bperm = RotLeft[Uperm[RotRight]]  # Bperm = RotRight<-Uperm<-RotLeft
        Lperm = RotFront[Uperm[RotBack]]  # Lperm = RotBack<-Uperm<-RotFront
        Dperm = RotRight[RotRight[Uperm[RotLeft[RotLeft]]]]  # Dperm = RotLeft<-RotLeft<-Uperm<-RotRight<-RotRight
        self.discrete_generators = [LazyPerm(perm) for perm in [Uperm,Fperm,Rperm,Bperm,Lperm,Dperm]]
        super().__init__()

@export
class ZksZnxZn(Group):
    """ One of the original GCNN groups ℤₖ⋉(ℤₙ×ℤₙ) for translation in x,y
        and rotation with the discrete 90 degree rotations (k=4) or 180 degree (k=2)"""
    def __init__(self,k,n):
        Zn = Z(n)
        Zk = Z(k)
        nshift = Zn.discrete_generators[0]
        kshift = Zk.discrete_generators[0]
        In = I(n)
        Ik = I(k)
        assert k in [2,4]
        self.discrete_generators = [LazyKron([Ik,nshift,In]),LazyKron([Ik,In,nshift]),LazyKron([kshift,Rot90(n,4//k)])]
        super().__init__(k,n)

@export
class Embed(Group):
    """ A method to embed a given base group representation in larger vector space.
    Inputs: 
    G: the group (and base representation) to embed
    d: the dimension in which to embed
    slice: a slice object specifying which dimensions G acts on."""
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
    """ SO(2) embedded in R^3 with rotations about z axis"""
    return Embed(SO(2),3,slice(2))

@export
def O2eR3():
    """ O(2) embedded in R^3 with rotations about z axis"""
    return Embed(O(2),3,slice(2))

@export
def DkeR3(k):
    """ Dihedral D(k) embedded in R^3 with rotations about z axis"""
    return Embed(D(k),3,slice(2))


class DirectProduct(Group):
    def __init__(self,G1,G2):
        I1,I2 = I(G1.d),I(G2.d)
        self.lie_algebra = [LazyKronsum([A1,0*I2]) for A1 in G1.lie_algebra]+[LazyKronsum([0*I1,A2]) for A2 in G2.lie_algebra]
        self.discrete_generators = [LazyKron([M1,I2]) for M1 in G1.discrete_generators]+[LazyKron([I1,M2]) for M2 in G2.discrete_generators]
        self.names = (repr(G1),repr(G2))
        super().__init__()

    def __repr__(self):
        return f"{self.names[0]}x{self.names[1]}"

class WreathProduct(Group):
    def __init__(self,G1,G2):
        raise NotImplementedError

class SemiDirectProduct(Group):
    def __init__(self,G1,G2,phi):
        raise NotImplementedError