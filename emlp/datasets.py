import numpy as np
import jax.numpy as jnp
from emlp.reps import Scalar,Vector,T
from emlp.utils import export,Named
from emlp.groups import SO,O,Trivial,Lorentz,RubiksCube,Cube
from functools import partial
import itertools
from jax import vmap,jit
from objax import Module

@export
class Inertia(object):
    def __init__(self,N=1024,k=5):
        super().__init__()
        self.dim = (1+3)*k
        self.X = np.random.randn(N,self.dim)
        self.X[:,:k] = np.log(1+np.exp(self.X[:,:k])) # Masses
        mi = self.X[:,:k]
        ri = self.X[:,k:].reshape(-1,k,3)
        I = np.eye(3)
        r2 = (ri**2).sum(-1)[...,None,None]
        inertia = (mi[:,:,None,None]*(r2*I - ri[...,None]*ri[...,None,:])).sum(1)
        self.Y = inertia.reshape(-1,9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        # One has to be careful computing offset and scale in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)
        Xmean[k:] = 0
        Xstd = np.zeros_like(Xmean)
        Xstd[:k] = np.abs(self.X[:,:k]).mean(0)#.std(0)
        #Xstd[k:] = (np.sqrt((self.X[:,k:].reshape(N,k,3)**2).mean((0,2))[:,None]) + np.zeros((k,3))).reshape(k*3)
        Xstd[k:] = (np.abs(self.X[:,k:].reshape(N,k,3)).mean((0,2))[:,None] + np.zeros((k,3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        #Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
        Ystd = np.abs(self.Y-Ymean).mean((0,1)) + np.zeros_like(Ymean)
        self.stats =0,1,0,1#Xmean,Xstd,Ymean,Ystd

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug(self,model):
        return GroupAugmentation(model,self.rep_in,self.rep_out,self.symmetry)

@export
class O5Synthetic(object):
    def __init__(self,N=1024):
        super().__init__()
        d=5
        self.dim = 2*d
        self.X = np.random.randn(N,self.dim)
        ri = self.X.reshape(-1,2,5)
        r1,r2 = ri.transpose(1,0,2)
        self.Y = np.sin(np.sqrt((r1**2).sum(-1)))-.5*np.sqrt((r2**2).sum(-1))**3 + (r1*r2).sum(-1)/(np.sqrt((r1**2).sum(-1))*np.sqrt((r2**2).sum(-1)))
        self.rep_in = 2*Vector
        self.rep_out = Scalar
        self.symmetry = O(d)
        self.Y = self.Y[...,None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0) # can add and subtract arbitrary tensors
        Xscale = (np.sqrt((self.X.reshape(N,2,d)**2).mean((0,2)))[:,None]+0*ri[0]).reshape(self.dim)
        self.stats = 0,Xscale,self.Y.mean(axis=0),self.Y.std(axis=0)

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug(self,model):
        return GroupAugmentation(model,self.rep_in,self.rep_out,self.symmetry)

@export
class ParticleInteraction(object):
    """ Electron muon e^4 interaction"""
    def __init__(self,N=1024):
        super().__init__()
        self.dim = 4*4
        self.rep_in = 4*Vector
        self.rep_out = Scalar
        self.X = np.random.randn(N,self.dim)/4
        P = self.X.reshape(N,4,4)
        p1,p2,p3,p4 = P.transpose(1,0,2)
        𝜂 = np.diag(np.array([1.,-1.,-1.,-1.]))
        dot = lambda v1,v2: ((v1@𝜂)*v2).sum(-1)
        Le = (p1[:,:,None]*p3[:,None,:] - (dot(p1,p3)-dot(p1,p1))[:,None,None]*𝜂)
        L𝜇 = ((p2@𝜂)[:,:,None]*(p4@𝜂)[:,None,:] - (dot(p2,p4)-dot(p2,p2))[:,None,None]*𝜂)
        M = 4*(Le*L𝜇).sum(-1).sum(-1)
        self.Y = M
        self.symmetry = Lorentz()
        self.Y = self.Y[...,None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        self.Xscale = np.sqrt((np.abs((self.X.reshape(N,4,4)@𝜂)*self.X.reshape(N,4,4)).mean(-1)).mean(0))
        self.Xscale = (self.Xscale[:,None]+np.zeros((4,4))).reshape(-1)
        self.stats = 0,self.Xscale,self.Y.mean(axis=0),self.Y.std(axis=0)#self.X.mean(axis=0),self.X.std(axis=0),
    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug(self,model):
        return GroupAugmentation(model,self.rep_in,self.rep_out,self.symmetry)

class GroupAugmentation(Module):
    def __init__(self,network,rep_in,rep_out,group):
        super().__init__()
        self.rep_in=rep_in
        self.rep_out=rep_out
        self.G=group
        self.rho_in = jit(vmap(self.rep_in.rho))
        self.rho_out = jit(vmap(self.rep_out.rho))
        self.model = network
    def __call__(self,x,training=True):
        if training:
            gs = self.G.samples(x.shape[0])
            rhout_inv = jnp.linalg.inv(self.rho_out(gs))
            return (rhout_inv@self.model((self.rho_in(gs)@x[...,None])[...,0],training)[...,None])[...,0]
        else:
            return self.model(x,False)

@export
class InvertedCube(object):
    def __init__(self,train=True):
        pass #TODO: finish implementing this simple dataset
        solved_state = np.eye(6)
        parity_perm = np.array([5,3,4,1,2,0])
        parity_state = solved_state[:,parity_perm]

        labels = np.array([1,0]).astype(int)
        self.X = np.zeros((2,6*6))
        self.X[0] = solved_state.reshape(-1)
        self.X[1] = parity_state.reshape(-1)
        self.Y = labels
        self.symmetry = Cube()
        self.rep_in = 6*Vector
        self.rep_out = 2*Scalar
        self.stats = (0,1)
        if train==False: # Scramble the cubes for test time
            gs = self.symmetry.samples(100)
            self.X = np.repeat(self.X,50,axis=0).reshape(100,6,6)@gs
            self.Y = np.repeat(self.Y,50,axis=0)
            p = np.random.permutation(100)
            self.X = self.X[p].reshape((100,-1))
            self.Y = self.Y[p].reshape((100,))
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
            
    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]


#### Ways of constructing invalid Rubik's cubes 
# (see https://ruwix.com/the-rubiks-cube/unsolvable-rubiks-cube-invalid-scramble/)

def UBedge_flip(state):
    """ # invalid edge flip on Rubiks state (6,48)"""
    UB_edge = np.array([1,3*8+1]) # Top middle of U, top middle of B
    edge_flip = state.copy()
    edge_flip[:,UB_edge] = edge_flip[:,np.roll(UB_edge,1)]
    return edge_flip

def ULBcorner_rot(state,i=1):
    """ Invalid rotated corner with ULB corner rotation """
    ULB_corner_ids = np.array([0,4*8,3*8+2]) # top left of U, top left of L, top right of B
    rotated_corner_state = state.copy()
    rotated_corner_state[:,ULB_corner_ids] = rotated_corner_state[:,np.roll(ULB_corner_ids,i)]
    return rotated_corner_state

def LBface_swap(state):
    """ Invalid piece swap between L center top and B center top """
    L_B_center_top_faces = np.array([4*8+1,3*8+1])
    piece_swap = state.copy()
    piece_swap[:,L_B_center_top_faces] = piece_swap[:,np.roll(L_B_center_top_faces,1)]
    return piece_swap


@export
class BrokenRubiksCube(object):
    """ Binary classification problem of predicting whether a Rubik's cube configuration
        is solvable or 'broken' and not able to be solved by transformations from the group
        e.g. by removing and twisting a corner before replacing.
        Dataset is generated by taking several hand identified simple instances of solvable
        and unsolvable cubes, and then scrambling them.
        
        Features are represented as 6xT(1) tensors of the Rubiks Group (one hot for each color)"""
    def __init__(self,train=True):
        super().__init__()
        # start with a valid configuration
        
        solved_state = np.zeros((6,48))
        for i in range(6):
            solved_state[i,8*i:8*(i+1)] = 1

        Id = lambda x: x
        transforms =  [Id,itertools.product([Id,ULBcorner_rot,partial(ULBcorner_rot,i=2)],
                                        [Id,UBedge_flip],
                                        [Id,LBface_swap])]
        #equivalence_classes = np.vstack([t3(t2(t1(solved_state))) for t1,t2,t3 in transforms])
        labels = np.zeros((22,))
        labels[:11]=1 # First configuration is solvable
        labels[11:]=0 # all others are not
        self.X = np.zeros((22,6*48)) # duplicate solvable example 11 times for convencience (balance)
        self.X[:11] = solved_state.reshape(-1)#equivalence_classes.reshape(12,-1)[:1]
        parity_perm = np.array([5,3,4,1,2,0])
        self.X[11:] = solved_state.reshape(6,6,8)[:,parity_perm].reshape(-1)#equivalence_classes.reshape(12,-1)[1:]
        self.Y = labels
        self.symmetry = RubiksCube()
        self.rep_in = 6*Vector
        self.rep_out = 2*Scalar
        self.stats = (0,1)
        if train==False: # Scramble the cubes for test time
            gs = self.symmetry.samples(440)
            self.X = np.repeat(self.X,20,axis=0).reshape(440,6,48)@gs
            self.Y = np.repeat(self.Y,20,axis=0)
            p = np.random.permutation(440)
            self.X = self.X[p].reshape((440,-1))
            self.Y = self.Y[p].reshape((440,))
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
            
    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]

# @export
# class BrokenRubiksCube2x2(object):
#     """ Binary classification problem of predicting whether a Rubik's cube configuration
#         is solvable or 'broken' and not able to be solved by transformations from the group
#         e.g. by removing and twisting a corner before replacing.
#         Dataset is generated by taking several hand identified simple instances of solvable
#         and unsolvable cubes, and then scrambling them.
        
#         Features are represented as 6xT(1) tensors of the Rubiks Group (one hot for each color)"""
#     def __init__(self,train=True):
#         super().__init__()
#         # start with a valid configuration
        
#         solved_state = np.zeros((6,24))
#         for i in range(6):
#             solved_state[i,4*i:4*(i+1)] = 1

#         ULB_corner_ids = np.array([0,4*4,3*4+2]) # top left of U, top left of L, top right of B
#         rotated_corner_state = solved_state.copy()
#         rotated_corner_state[:,ULB_corner_ids] = rotated_corner_state[:,np.roll(ULB_corner_ids,1)]
        
#         labels = np.zeros((2,))
#         labels[:1]=1 # First configuration is solvable
#         labels[1:]=0 # all others are not
#         self.X = np.zeros((2,6*24)) # duplicate solvable example 11 times for convencience (balance)
#         self.X[0] = solved_state.reshape(-1)
#         parity_perm = np.array([5,3,4,1,2,0])
#         self.X[1] = rotated_corner_state.reshape(6,6,4)[:,parity_perm].reshape(-1)
#         self.Y = labels
#         self.X = np.repeat(self.X,10,axis=0)
#         self.Y = np.repeat(self.Y,10,axis=0)
#         self.symmetry = RubiksCube2x2()
#         self.rep_in = 6*Vector
#         self.rep_out = 2*Scalar
#         self.stats = (0,1)
#         if train==False: # Scramble the cubes for test time
#             N = 200
#             gs = self.symmetry.samples(N)
#             self.X = np.repeat(self.X,10,axis=0).reshape(N,6,24)@gs
#             self.Y = np.repeat(self.Y,10,axis=0)
#             p = np.random.permutation(N)
#             self.X = self.X[p].reshape((N,-1))
#             self.Y = self.Y[p].reshape((N,))
#         self.X = np.array(self.X)
#         self.Y = np.array(self.Y)
            
#     def __getitem__(self,i):
#         return (self.X[i],self.Y[i])
#     def __len__(self):
#         return self.X.shape[0]