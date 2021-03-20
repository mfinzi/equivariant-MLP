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

# Multilinear Maps

+++

Our codebase extends trivially to multilinear maps, since these maps are in fact just linear maps in disguise.

If we have a sequence of representations $R_1$, $R_2$, $R_3$ for example, we can write the (bi)linear maps $R_1\rightarrow R_2\rightarrow R_3$. This way of thinking about maps of multiple variables borrowed from programming languages and curried functions is very powerful.

+++

We can think of such an object $R_1\rightarrow R_2\rightarrow R_3$ either as $R_1 \rightarrow (R_2\rightarrow R_3)$: a linear map from $R_1$ to linear maps from $R_2$ to $R_3$ or as
$(R_1\times R_2) \rightarrow R_3$: a bilinear map from $R_1$ and $R_2$ to $R_3$. Since linear maps from one representation to another are just another representation in our type system, you can use this way of thinking to find the equivariant solutions to arbitrary multilinear maps.

+++

For example, we can get the bilinear $SO(4)$ equivariant maps $(R_1\times R_2) \rightarrow R_3$ with the code below.

```{code-cell} ipython3
from emlp.solver.groups import SO,rel_err
from emlp.solver.representation import V

G = SO(4)
W = V(G)
R1 = 3*W+W**2 # some example representations
R2 = W.T+W**0
R3 = W**0 +W**2 +W

Q = (R1>>(R2>>R3)).symmetric_basis()
print(Q.shape)
```

And we can verify that these multilinear solutions are indeed equivariant

```{code-cell} ipython3
import numpy as np

example_map = (Q@np.random.randn(Q.shape[-1]))
example_map = example_map.reshape(R3.size(),R2.size(),R1.size())

x1 = np.random.randn(R1.size())
x2 = np.random.randn(R2.size())
g = G.sample()

out1 = np.einsum("ijk,j,k",example_map,R2.rho(g)@x2,R1.rho(g)@x1)
out2 = R3.rho(g)@np.einsum("ijk,j,k",example_map,x2,x1)
rel_err(out1,out2)
```

Note that the output mapping is of shape $(\mathrm{dim}(R_3),\mathrm{dim}(R_2),\mathrm{dim}(R_1))$
with the inputs to the right as you would expect with a matrix. 

Note the parenthesis in the expression `(R1>>(R2>>R3))` since the python `>>` associates to the right.
The notation $R_1\rightarrow R_2 \rightarrow R_3$ or `(R1>>(R2>>R3))` can be a bit confusing since the inputs are on the right. It can be easier in this concept to instead reverse the arrows and express the same object as $R_3\leftarrow R_2\leftarrow R_1$ or `R3<<R2<<R1` (with no parens required) that matches the axis ordering of the multilinear map (tensor). 

You can use `R2<<R1` in place of `R1>>R2` wherever you like, and it is usually more intuitive.

```{code-cell} ipython3
R3<<R2<<R1 == (R1>>(R2>>R3))
```
