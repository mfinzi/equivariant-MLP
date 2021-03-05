---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Constructing Equivariant Models

+++

Previously we showed examples of finding equivariant bases for different groups and representations, now we'll show how these bases can be assembled into equivariant neural networks such as EMLP. 

We will give examples at a high level showing how the specific EMLP model can be applied to different groups and input-output types, and later in a lower level showing how models like EMLP can be constructed with equivariant layers and making use of the equivariant bases.

+++

## Using EMLP with different groups and representations (high level)

+++

![ex 2.13](imgs/EMLP_fig.png)

+++

A basic EMLP is a sequence of EMLP layers (containing G-equivariant linear layers, bilinear layers incorporated with a shortcut connection, and gated nonlinearities. While our numerical equivariance solver can work with any finite dimensional linear representation, for EMLP we restrict ourselves to _tensor_ representations.

+++

By tensor representations, we mean all representations which can be formed by arbitrary combinations of $\oplus$,$\otimes$,$^*$ (`+`,`*`,`.T`) of a base representation $\rho$. This is useful because it simplifies the construction of our bilinear layer, which is a crucial ingredient for expressiveness and universality in EMLP.

Following the $T_{(p,q)}=V^{\otimes p}\otimes (V^*)^{\otimes p}$ notation in the paper, we provide the convenience function for constructing higher rank tensors.

```{code-cell} ipython3
from emlp.solver.representation import V
from emlp.solver.groups import SO13

def T(p,q=0):
    return (V**p*V.T**q)

print(T(2,3))
print(T(2,3)(SO13()))
```

Lets get started with a toy dataset: learning how an inertia matrix depends on the positions and masses of 5 point masses distributed in different ways. The data consists of mappings (positions, masses) --> (inertia matrix) pairs, and has an $G=O(3)$ symmetry (3D rotation and reflections). If we rotate all the positions, the resulting inertia matrix should be correspondingly rotated.

```{code-cell} ipython3
from emlp.models.datasets import Inertia
from emlp.solver.groups import SO,O,S,Z
trainset = Inertia(1000) # Initialize dataset with 1000 examples
testset = Inertia(2000)
G = SO(3)
print(f"Input type: {trainset.rep_in(G)}, output type: {trainset.rep_out(G)}")
```

For convenience, we store in the dataset the types for the input and the output. `5V⁰` are the $5$ mass values and `5V` are the position vectors of those masses, `V²` is the matrix type for the output, equivalent to $T_2$. To initialize the EMLP, we just need these input and output representations, the symmetry group, and the size of the network as parametrized by number of layers and number of channels (the dimension of the feature representation).

```{code-cell} ipython3
from emlp.models.mlp import EMLP,MLP
model = EMLP(trainset.rep_in,trainset.rep_out,group=G,num_layers=3,ch=384)
# uncomment the following line to instead try the MLP baseline
#model = MLP(trainset.rep_in,trainset.rep_out,group=G,num_layers=3,ch=384)
```

## Example Objax Training Loop

+++

We build our EMLP model with [objax](https://objax.readthedocs.io/en/latest/) because we feel the object oriented design makes building complicated layers easier. Below is a minimal training loop that you could use to train EMLP.

```{code-cell} ipython3
BS=500
lr=3e-3
NUM_EPOCHS=500

import objax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from jax import vmap

opt = objax.optimizer.Adam(model.vars())

@objax.Function.with_vars(model.vars())
def loss(x, y,training=True):
    yhat = model(x, training=training)
    return ((yhat-y)**2).mean()

gv = objax.GradValues(loss, model.vars())

@objax.Function.with_vars(model.vars()+opt.vars())
def train_op(x, y, lr):
    g, v = gv(x, y)
    opt(lr=lr, grads=g)
    return v

train_op = objax.Jit(train_op)
test_loss = objax.Jit(objax.ForceArgs(loss, training=False))

trainloader = DataLoader(trainset,batch_size=BS,shuffle=True)
testloader = DataLoader(testset,batch_size=BS,shuffle=True)
```

```{code-cell} ipython3
test_losses = []
train_losses = []
for epoch in tqdm(range(NUM_EPOCHS)):
    train_losses.append(np.mean([train_op(jnp.array(x),jnp.array(y),lr) for (x,y) in trainloader]))
    if not epoch%10:
        test_losses.append(np.mean([test_loss(jnp.array(x),jnp.array(y)) for (x,y) in testloader]))
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.plot(np.arange(NUM_EPOCHS),train_losses,label='Train loss')
plt.plot(np.arange(0,NUM_EPOCHS,10),test_losses,label='Test loss')
plt.legend()
plt.yscale('log')
```

```{code-cell} ipython3
def rel_err(a,b):
    return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

rin,rout = trainset.rep_in(G),trainset.rep_out(G)

def equivariance_err(mb):
    x,y = mb
    x,y= jnp.array(x),jnp.array(y)
    gs = G.samples(x.shape[0])
    rho_gin = vmap(rin.rho_dense)(gs)
    rho_gout = vmap(rout.rho_dense)(gs)
    y1 = model((rho_gin@x[...,None])[...,0],training=False)
    y2 = (rho_gout@model(x,training=False)[...,None])[...,0]
    return rel_err(y1,y2)
```

As expected, the network continues to be equivariant as it is trained.

```{code-cell} ipython3
print(f"Average test equivariance error {np.mean([equivariance_err(mb) for mb in testloader]):.2e}")
```

## Equivariant Linear Layers (low level) 

+++

Internally for EMLP, we use representations that uniformly allocate dimensions between different tensor representations.

```{code-cell} ipython3
from emlp.models.mlp import uniform_rep
r = uniform_rep(512,G)
print(r)
```

Below is a trimmed down version of EMLP, so you can see how it is built from the component layers.

```{code-cell} ipython3
from objax.module import Module
from emlp.models.mlp import Sequential,EMLPBlock,LieLinear

class EMLP(Module):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        reps = [rep_in(group)]+num_layers*[uniform_rep(ch,group)]
        self.network = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            LieLinear(reps[-1],rep_out(group))
        )
    def __call__(self,x,training=True):
        return self.network(x)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
# from emlp.models.mlp import Standardize
# from emlp.models.model_trainer import RegressorPlus
# from emlp.slax.utils import LoaderTo
# BS=500
# lr=3e-3
# NUM_EPOCHS=100

# dataloaders = {k:LoaderTo(DataLoader(v,batch_size=BS,shuffle=(k=='train'),
#                 num_workers=0,pin_memory=False)) for k,v in {'train':trainset,'test':testset}.items()}
# dataloaders['Train'] = dataloaders['train']
# opt_constr = objax.optimizer.Adam
# lr_sched = lambda e: lr
# trainer = RegressorPlus(model,dataloaders,opt_constr,lr_sched,log_args={'minPeriod':.02,'timeFrac':.25})
# trainer.train(NUM_EPOCHS)
```
