
import numpy as np#
import copy
from emlp.equivariant_subspaces import *
import unittest

def rel_error(t1,t2):
    return np.mean(np.abs(t1-t2))/(np.mean(np.abs(t1)) + np.mean(np.abs(t2))+1e-7)

def rot(angle):
    R =np.zeros((2,2))
    R[...,0,0] = R[...,1,1] = np.cos(angle)
    R[...,0,1] = np.sin(angle)
    R[...,1,0] = - np.sin(angle)
    return R

def rep_act_on_tensor(G,tensor,rank):
    p,q = rank
    Ginv = np.linalg.inv(G)
    tensor_out = copy.deepcopy(tensor)
    batch_dims = len(tensor.shape[:-(p+q)])
    for i in range(p):
        tensor_out = (tensor_out.swapaxes(-1,batch_dims+i)@G.T).swapaxes(-1,batch_dims+i)
    for j in range(q):
        tensor_out = (tensor_out.swapaxes(-1,batch_dims+p+j)@Ginv).swapaxes(-1,batch_dims+p+j)
    return tensor_out

def rep_act_on_representation(G,rep_vector,ranks,d=2):
    batch_shape = rep_vector.shape[:-1]
    vector_out = np.zeros_like(rep_vector)
    i=0
    for (p,q) in ranks:
        size = d**(p+q)
        tensor = rep_vector[...,i:i+size].reshape(*batch_shape,(d,)*(p+q))
        vector_out[...,i:i+size] = rep_act_on_tensor(G,tensor,(p,q)).reshape(*batch_shape,-1)
        i+=size
    return vector_out

class TestRepresentationSubspace(unittest.TestCase):
    def test_composite_representation(self):
        rxy = np.array([[0,1,0],[-1,0,0],[0,0,0]])
        gens = (rxy,)
        ranks = T(0,0)+T(1,0)+T(0,0)+T(1,1)+T(1,0)
        active_dims, proj = ranks(gens).symmetric_subspace()#get_active_subspaces(gens,ranks)
        random_rep_vec = proj(torch.randn(active_dims)).data.numpy()
        self.assertTrue(rel_error(random_rep_vec[1:3],np.zeros(2))<1e-7)
        self.assertTrue(rel_error(random_rep_vec[-3:-1],np.zeros(2))<1e-7)
        A = random_rep_vec[5:5+3*3].reshape(3,3)
        commutator = A@rxy-rxy@A
        self.assertTrue(rel_error(commutator,np.zeros((3,3)))<1e-7)

    def test_equivariant_matrix_subspace(self):
        repin = 2*Scalar+3*Vector+T(1,2) + Matrix+Vector
        repout = T(2,1)+2*Vector+ 3*Scalar+Quad
        gens = [np.array([[0,-1],[1,0]])]
        active_dims,P = (repout*repin.T)(gens).symmetric_subspace()#matrix_active_subspaces(gens,repout,repin)
        params = torch.randn(active_dims).cuda()
        W = P(params).cpu()
        x = torch.randn(repsize(repin,2))
        Wx = (x.unsqueeze(0)@W.T).squeeze(0)
        for angle in np.linspace(-np.pi,np.pi,10):
            R = rot(angle)
            pWx = rep_act_on_representation(R,Wx.numpy(),repout)
            Wpx = (rep_act_on_representation(R,x.numpy(),repin)[None]@W.T.numpy())[0]
            self.assertTrue(rel_error(pWx,Wpx)<1e-7)

if __name__ == '__main__':
    unittest.main()