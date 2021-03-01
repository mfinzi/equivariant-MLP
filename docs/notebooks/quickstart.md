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

# Getting Familiar with the Type System

```{code-cell} ipython3
from emlp.solver.representation import V,sparsify_basis
from emlp.solver.groups import *
import jax.numpy as jnp
import logging
logging.getLogger().setLevel(logging.INFO)
```

EMLP computes the symmetric subspace for a linear representation $\rho$ and a matrix group $G$, solving the constraint to find an element $v\in V$ that satisfies$$\forall g\in G: \ \ \rho(g)v=v$$

For example, we can find invariant vectors of the cyclic translation group $\mathbb{Z}_n$ which is just the constant $\vec{1}$ scaled to have unit norm. 

```{code-cell} ipython3
V(Z(5)).symmetric_basis()
```

Each implemented group comes with a faithful 'base' representation $V$. Because faithful representations are one-to-one, we can build any representation by transforming this base representation.

We provide several operators to transform and construct representations in different ways built and later go on to show how to do this more generally. In our type system, representations can be combined with the direct sum $\rho_a \oplus\rho_b$ operator, the tensor product $\rho_a\otimes\rho_b$, the dual $\rho^*$. We implement these with the python operators `+`, `*`, and `.T`.

```{code-cell} ipython3
V+V,  V*V,  V.T
```

We can combine and use these operators interchangeably:

```{code-cell} ipython3
(V+V.T)*(V*V.T+V)
```

We use the shorthand $cV$ can be used for $V\oplus V\oplus...\oplus V$ and $V^c = V\otimes V\otimes...\otimes V$, not that this different from the typical notation with cartesian products of sets.

```{code-cell} ipython3
5*V*2

2*V**3
```

When a particular symmetry group is specified, the representation can be collapsed down to a more compact form:

```{code-cell} ipython3
G=O(4)
2*V(G)**3
```

```{code-cell} ipython3
(2*V**3)(G)
```

Although for groups like the Lorentz group $SO(1,3)$ with non orthogonal representations, a distinction needs to be made between the representation and it's dual. In both cases the representation is converted down to a canonical form (but the ordering you gave is preserved as a permutation).

```{code-cell} ipython3
V(SO(3)).T+V(SO(3))

V(SO13()).T+V(SO13())
```

Linear maps from $V_1\rightarrow V_2$ have the type $V_2\otimes V_1^*$. The `V>>W` is shorthand for `W*V.T` and produces linear maps from `V` to `W`.

Imposing (cyclic) Translation Equivariance $G=\mathbb{Z}_n$ on linear maps $V\rightarrow V$ yields circular convolutions (circulant matrices) which can be expressed as a linear combination of $n$ basis elements of size $n\times n$.

```{code-cell} ipython3
G = Z(6)
repin = V(G)
repout = V(G)
conv_basis = (repin>>repout).symmetric_basis()
print(conv_basis.shape)
```

While we provide an orthogonal basis, these bases are not always easy to make sense of as an array of numbers (any rotation of an orthogonal basis is still an orthogonal basis)

```{code-cell} ipython3
conv_basis[:,0]
```

To more easily visualize the result, we can define the following function which projects a random vector and then plots components with the same values as different colors, arranged in a desired shape.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def vis_basis(basis,shape,cluster=True):
    Q=basis@jnp.eye(basis.shape[-1]) # convert to a dense matrix if necessary
    v = np.random.randn(Q.shape[0])  # sample random vector
    v = Q@(Q.T@v)                    # project onto equivariant subspace
    if cluster: # cluster nearby values for better color separation in plot
        v = KMeans(n_clusters=Q.shape[-1]).fit(v.reshape(-1,1)).labels_
    plt.imshow(v.reshape(shape))
    plt.axis('off')

def vis(repin,repout,cluster=True):
    Q = (repin>>repout).symmetric_basis()
    vis_basis(Q,(repout.size(),repin.size()),cluster)
```

Our convolutional basis is the familiar (circulant) convolution matrix.

```{code-cell} ipython3
vis_basis(conv_basis,(repin.size(),repout.size()))
```

2D translation equivariange $G=\mathbb{Z}_n\times \mathbb{Z}_n$ yields 2D convolutions (bicurculant matrices)

```{code-cell} ipython3
G = Z(3)*Z(4) #[Not recommended way of building product groups, there is a faster path described later]
repin = V(G)
repout = V(G)
vis(repin,repout)
```

How about the permutation group $G=S_n$, where the vector space $V$ represents a set of elements? In deep sets it was shown there are only two basis elements for equivariant linear maps $V\rightarrow V$.

```{code-cell} ipython3
repin = V(S(6))
repout = V(S(6))
vis(repin,repout)
```

What about graphs, which are composed both of sets as well as adjacency matrices or graph laplacians? These matrices are examples of objects from $V\otimes V$ with $G=S_n$, and in Invariant and Equivariant Graph Networks () it was shown through a challenging proof that there are at most 15 basis elements which were derived analytically. We can solve for them here:

```{code-cell} ipython3
repin = V(S(6))**2
repout = V(S(6))**2
vis(repin,repout)
print((repin>>repout).symmetric_basis().shape)
```

## Composite Representations

+++

How about maps from graphs to sets? Lets say a graph consists of one node feature and one edge feature which can be represented with the $\oplus$ operator.

```{code-cell} ipython3
W = V(S(6))
repin = W+W**2 # (one set feature and one edge feature)
repout = W     # (one set feature)

vis(repin,repout)
print((repin>>repout).symmetric_basis().shape)
```

We can compute the bases for representations that have many copies or multiplicity of a given representation type, such as for the many channels in a neural network. The `rep.symmetric_basis()` and `rep.symmetric_projector()` can return lazy matrices $Q$ and $P=QQ^T$ when the representations are composite (or when the representation is specified lazily). [implementation change is making this much slower than normal, using smaller values]

+++

For example with a more realistically sized layer with 100 global constants, 100 set feature channels, and 20 edge feature channels ($100V^0+100V^1+20V^2$) we have

```{code-cell} ipython3

W = V(S(6))
repin = 100*W**0 + 100*W+20*W**2
repout = repin
rep_map = repin>>repout
print(f"{rep_map}, of size {rep_map.size()}")

# Q = rep_map.symmetric_basis()
# print(Q.shape)
```

These Lazy matrices are modeled after https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html. Unfortunately the larger matrices are harder to visualize, maybe someone can figure out how to do this better?

```{code-cell} ipython3
P =rep_map.symmetric_projector()
v = np.random.randn(P.shape[-1])
v = P@v
plt.imshow(v.reshape(repout.size(),repin.size()))
plt.axis('off')
```

But more fun than continuous groups are discrete groups!
How about the $2$D rotation group $SO(3)$. It's well known that the only equivariant object for the vector space $V^{\otimes 3}$ is the Levi-Civita symbol $\epsilon_{ijk}$. Since the values are both $0$, positive, and negative (leading to more than `Q.shape[-1]` clusters) we disable the clustering.

```{code-cell} ipython3
W = V(SO(3))
repin = W**2
repout = W
Q = (repin>>repout).symmetric_basis()
print(Q.shape)
vis(repin,repout,cluster=False)

print(sparsify_basis(Q).reshape(3,3,3))
```

```{code-cell} ipython3
from emlp.solver.representation import T
```

## High Dimensional Representations

+++

We can also solve for very high dimensional representations which we automatically switch to using the automated iterative Krylov subspace method

```{code-cell} ipython3
vis(W**3,W**3)
```

```{code-cell} ipython3
vis(W**5,W**3)
```

```{code-cell} ipython3
vis(V(RubiksCube()),V(RubiksCube()))
```
