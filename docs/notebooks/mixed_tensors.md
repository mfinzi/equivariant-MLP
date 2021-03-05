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

# Combining Representations from Different Groups (experimental)

```{code-cell} ipython3
from emlp.solver.groups import *
from emlp.solver.representation import T,vis
```

```{code-cell} ipython3
rep = 2*T(1)(Z(3))*T(1)(S(4))+T(1)(SO(2))
```

```{code-cell} ipython3
(rep>>rep)
```

```{code-cell} ipython3
vis(rep,rep)
```

```{code-cell} ipython3
repin,repout = T(1)(SO(3))*T(2)(S(4)),T(2)(SO(3))*T(1)(S(4))
```

```{code-cell} ipython3
repin>>repout
```

```{code-cell} ipython3
#vis(repin,repout)
```

```{code-cell} ipython3

```
