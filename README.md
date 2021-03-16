# A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups

<!-- 
.. image:: https://img.shields.io/pypi/v/<name-of-package>.svg
   :target: https://pypi.python.org/pypi/<name-of-package>
   :alt: PyPi version -->
--------------------------------------------------------------------------------
[![Documentation](https://readthedocs.org/projects/emlp/badge/)](https://emlp.readthedocs.io/en/latest/) | **[Paper](https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstleyVEVO)**| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfinzi/equivariant-MLP/blob/master/emlp.ipynb)


*EMLP* is a jax library for the automated construction of equivariant layers in deep learning.

*WARNING*: Our library (and paper) have not yet been released, and may have sharp edges, bugs, and may be subject to breaking changes. 
Use at your own caution. But if you notice things behaving unexpectedly or get frustrated, send me an email so I can make the library better.

--------------------------------------------------------------------------------

Our type system is centered on it making it easy to combine representations using ρᵤ⊗ρᵥ, ρᵤ⊕ρᵥ, ρ*. For any given matrix group and representation formed in our type system, you can get the equivariant basis with `rep.symmetric_basis()` or a matrix which projects to that subspace with `rep.symmetric_projector()`. For example:

```python
from emlp.solver.representation import V
from emlp.solver.groups import *

W=V(O13())
repin = (W+2*W**2)*(W.T+1*W).T + W.T
repout = 3*W**0 + W + W*W.T
Q = (repin>>repout).symmetric_basis()
```

is code that will run and produce the basis for linear maps from repin to repout that are equivariant to the Lorentz group O(1,3).

You can even mix and match representations from different groups. For example with the cyclic group Z, the permutation group S, and the orthogonal group O

```python
rep = 2*V(Z(3))*V(S(4))+V(O(3))**2
Q = (rep>>rep).symmetric_basis()
```

You can visualize these equivariant bases with `vis(repin,repout)`, such as with the two examples above

<img src="https://user-images.githubusercontent.com/12687085/111226517-a2192b80-85b7-11eb-8dba-c01399fb7105.png" width="400"/> <img src="https://user-images.githubusercontent.com/12687085/111226510-a0e7fe80-85b7-11eb-913b-09776cdaa92e.png" width="250"/>  
<!-- ![basis B](https://user-images.githubusercontent.com/12687085/111226517-a2192b80-85b7-11eb-8dba-c01399fb7105.png "title2")
![basis A](https://user-images.githubusercontent.com/12687085/111226510-a0e7fe80-85b7-11eb-913b-09776cdaa92e.png "title1") -->


Checkout our [documentation](https://emlp.readthedocs.io/en/latest/) how to use our system and some worked examples.


# Installation instructions

To install as a package, run `pip install git+https://github.com/mfinzi/equivariant-MLP.git`.

To run the scripts you will instead need to clone the repo and install it locally which you can do with
```bash
git clone https://github.com/mfinzi/equivariant-MLP.git
cd equivariant-MLP
pip install -e .
```

# Experimental Results from Paper

Assuming you have installed the repo locally, you can run the experiments we described in the paper. The relevant scripts are
[`emlp/experiments/neuralode.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/experiments/neuralode.py) to train (equivariant) Neural ODEs, [`emlp/experiments/hnn.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/experiments/hnn.py) to train (equivariant) Hamiltonian Neural Networks, [`emlp/experiments/inertia.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/experiments/inertia.py) to train (E)MLP on the inertia task, [`emlp/experiments/o5_synthetic.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/experiments/o5_synthetic.py) for the O(5)-Invariant synthetic task, and [`emlp/experiments/scattering.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/experiments/scattering.py) to train (E)MLP on the Lorentz invariant scattering task. If in doubt, you can see options for command line arguments for each of these scripts by running `python script.py -h`.

For the three regression tasks, you can run EMLP or the baseline with `--network EMLP` or `--network MLP` to train the baseline. For example:

```
python emlp/experiments/scattering.py --network EMLP
```

For the dynamical system task, the Neural ODE and HNN models have special names. `EMLPode` and `MLPode` for the Neural ODEs in `neuralode.py` and `EMLPH` and `MLPH` for the HNNs in `hnn.py`. For example,
```
python emlp/experiments/neuralode.py --network EMLPode
```

You can also train EMLP using other equivariance groups provided the dimensions match the data. To do this, specify a group from [`groups.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/solver/groups.py) in quotations, for example you could swap out SO(3) in `inertia.py` with the permutation group S(3) using the command line argument `--group "S(3)"`. However, other groups will likely be less relevant to these specific problems.

<!-- # 
<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94081992-e75d5d00-fdcd-11ea-9df0-576af6909944.PNG" width=1000>
</p> -->

If you find our work helpful, please cite it with
```bibtex
@article{finzi2021emlp,
  title={A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups},
  author={Finzi, Marc and Welling, Max and Wilson, Andrew Gordon},
  journal={Arxiv},
  year={2021}
}
```
<!-- 
Top quark tagging dataset: https://zenodo.org/record/2603256#.YAoEPehKiUl -->