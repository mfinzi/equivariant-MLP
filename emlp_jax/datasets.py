import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from emlp_jax.equivariant_subspaces import Scalar,Vector,Matrix,T
from torch.utils.data import Dataset
from oil.utils.utils import export,Named,Expression,FixedNumpySeed
from emlp.groups import SO,O,Trivial,Lorentz

@export
class Inertia(Dataset,metaclass=Named):
    def __init__(self,N=1024,k=5):
        super().__init__()
        self.dim = (1+3)*k
        self.X = torch.randn(N,self.dim)
        self.X[:,:k] = F.softplus(self.X[:,:k]) # Masses
        mi = self.X[:,:k]
        ri = self.X[:,k:].reshape(-1,k,3)
        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[...,None,None]
        inertia = (mi[:,:,None,None]*(r2*I - ri[...,None]*ri[...,None,:])).sum(1)
        self.Y = inertia.reshape(-1,9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = Matrix
        self.symmetry = O(3)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()
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
    def default_aug_layers(self):
        return nn.Sequential()

@export
class Fr(Dataset,metaclass=Named):
    def __init__(self,N=1024):
        super().__init__()
        d=3
        self.dim = 2*d
        self.X = torch.randn(N,self.dim)
        ri = self.X.reshape(-1,2,d)
        self.Y = (ri[:,0]**2).sum(-1).sqrt().sin()-.5*(ri[:,1]**2).sum(-1).sqrt()**3
        self.rep_in = 2*Vector
        self.rep_out = Scalar
        self.symmetry = O(d)
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()[...,None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0) # can add and subtract arbitrary tensors
        Xscale = (np.sqrt((self.X.reshape(N,2,d)**2).mean((0,2)))[:,None]+0*ri[0].numpy()).reshape(self.dim)
        self.stats = 0,Xscale,self.Y.mean(axis=0),self.Y.std(axis=0)

    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug_layers(self):
        return GroupAugmentation(self.rep_in,self.symmetry)

@export
class ParticleInteraction(Dataset,metaclass=Named):
    """ Electron muon e^4 interaction"""
    def __init__(self,N=1024):
        super().__init__()
        self.dim = 4*4
        self.rep_in = 4*Vector
        self.rep_out = Scalar
        self.X = torch.randn(N,self.dim)/4
        P = self.X.reshape(N,4,4)
        p1,p2,p3,p4 = P.permute(1,0,2)
        𝜂 = torch.diag(torch.tensor([1.,-1.,-1.,-1.]))
        dot = lambda v1,v2: ((v1@𝜂)*v2).sum(-1)
        Le = (p1[:,:,None]*p3[:,None,:] - (dot(p1,p3)-dot(p1,p1))[:,None,None]*𝜂)
        L𝜇 = ((p2@𝜂)[:,:,None]*(p4@𝜂)[:,None,:] - (dot(p2,p4)-dot(p2,p2))[:,None,None]*𝜂)
        M = 4*(Le*L𝜇).sum(-1).sum(-1)
        self.Y = M
        self.symmetry = Lorentz
        self.X = self.X.numpy()
        self.Y = self.Y.numpy()[...,None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        self.Xscale = np.sqrt((np.abs((self.X.reshape(N,4,4)@𝜂.numpy())*self.X.reshape(N,4,4)).mean(-1)).mean(0))
        self.Xscale = (self.Xscale[:,None]+np.zeros((4,4))).reshape(-1)
        self.stats = 0,self.Xscale,self.Y.mean(axis=0),self.Y.std(axis=0)#self.X.mean(axis=0),self.X.std(axis=0),
    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug_layers(self):
        return GroupAugmentation(self.rep_in,self.symmetry)

class GroupAugmentation(nn.Module):
    def __init__(self,rep,group):
        super().__init__()
        self.rep=rep
        self.group=group
    def forward(self,x):
        if self.training:
            rhog = np.stack([self.rep.rho(g) for g in self.group.samples(x.shape[0])])
            return  (rhog@x[...,None])[...,0]
        else:
            return x