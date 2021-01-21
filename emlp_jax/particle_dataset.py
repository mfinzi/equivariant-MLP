
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from emlp_jax.equivariant_subspaces import Scalar,Vector,Matrix,T
from torch.utils.data import Dataset
from oil.utils.utils import export,Named,Expression,FixedNumpySeed
from emlp.groups import SO,O,Trivial,Lorentz
import pandas as pd
import os
import numpy as np
import torch

def _col_list(prefix, max_particles=200):
    return ['%s_%d'%(prefix,i) for i in range(max_particles)]

@export
class TopTagging(Dataset,metaclass=Named):
    def __init__(self,split='train'):
        super().__init__()
        self.dim = 200*4
        self.rep_in = 200*Vector
        self.rep_out = Scalar
        self.symmetry = Lorentz
        df = pd.read_hdf(os.path.expanduser("~/datasets/top/")+f"{split}.h5", key='table')
        self.X = np.stack([df[_col_list(a)].values for a in ['PX','PY','PZ','E']],axis=-1) #(B,200,4)
        self.mask = self.X[:,:,-1]>0
        self.Y = df['is_signal_new'].values>0
        b,n,c = self.X.shape
        𝜂 = np.diag(np.array([1.,-1.,-1.,-1.]))
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        #self.Xscale = np.sqrt(((self.X@𝜂)*self.X).sum(-1)).sum()/self.mask.sum()
        self.Xscale = self.X.std()
        print(self.Xscale)
    def __getitem__(self,i):
        return ((self.X[i]/self.Xscale,self.mask[i]),self.Y[i])
    def __len__(self):
        return self.X.shape[0]
    # def default_aug_layers(self):
    #     return GroupAugmentation(self.rep_in,self.symmetry)

def collate_fn(batch):
    (X,mask) = np.stack([b[0][0] for b in batch]), np.stack([b[0][1] for b in batch])
    Y = np.stack([b[1] for b in batch])
    Ns = mask.astype(int).sum(-1).max(0)
    Ns = (Ns//10 )*10 + (10 if Ns%10 else 0) # Round up to nearest 10
    X = X[:,:Ns]#mask.sum(0)>0]
    mask = mask[:,:Ns]#mask.sum(0)>0]
    #print(X.shape,mask.shape)
    return (X,mask),Y