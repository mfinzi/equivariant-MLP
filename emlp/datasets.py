import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from emlp.equivariant_subspaces import Scalar,Vector,Matrix
from torch.utils.data import Dataset
from oil.utils.utils import export,Named,Expression,FixedNumpySeed

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
        inertia = (mi[:,:,None,None]*((ri**2).sum(-1)[...,None,None]*I - ri[...,None]*ri[...,None,:])).sum(1)
        self.Y = inertia.reshape(-1,9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = Matrix



    def __getitem__(self,i):
        return (self.X[i],self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    def default_aug_layers(self):
        return nn.Sequential()