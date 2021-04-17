<div align="center">
<img src="docs/_static/emlp_logo4x.png" width="350" alt="logo"/>
</div>

# A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups
[![Documentation](https://readthedocs.org/projects/emlp/badge/)](https://emlp.readthedocs.io/en/latest/) | **[Paper]()**| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfinzi/equivariant-MLP/blob/master/docs/notebooks/colabs/all.ipynb) | ![PyPI version](https://img.shields.io/pypi/v/emlp)


*EMLP* is a jax library for the automated construction of equivariant layers in deep learning. You can read the documentation [here](https://emlp.readthedocs.io/en/latest/).

*WARNING*: Our library (and paper) have not yet been released, and may have sharp edges, bugs, and may be subject to breaking changes. 
Use at your own caution. But if you notice things behaving unexpectedly or get frustrated, send me an email so I can make the library better.

--------------------------------------------------------------------------------

Our type system is centered on it making it easy to combine representations using ρᵤ⊗ρᵥ, ρᵤ⊕ρᵥ, ρ*. For any given matrix group and representation formed in our type system, you can get the equivariant basis with [`rep.equivariant_basis()`](https://emlp.readthedocs.io/en/latest/package/emlp.reps.html#emlp.reps.equivariant_basis) or a matrix which projects to that subspace with [`rep.equivariant_projector()`](https://emlp.readthedocs.io/en/latest/package/emlp.reps.html#emlp.reps.equivariant_projector). For example:

```python
from emlp.reps import V
from emlp.groups import *

W=V(O13())
repin = (W+2*W**2)*(W.T+1*W).T + W.T
repout = 3*W**0 + W + W*W.T
Q = (repin>>repout).equivariant_basis()
```

is code that will run and produce the basis for linear maps from repin to repout that are equivariant to the Lorentz group O(1,3).

You can even mix and match representations from different groups. For example with the cyclic group ℤ₃, the permutation group 𝕊₄, and the orthogonal group O(3)

```python
rep = 2*V(Z(3))*V(S(4))+V(O(3))**2
Q = (rep>>rep).equivariant_basis()
```

You can visualize these equivariant bases with [`vis(repin,repout)`](https://emlp.readthedocs.io/en/latest/package/emlp.reps.html#emlp.reps.vis), such as with the two examples above

<img src="https://user-images.githubusercontent.com/12687085/111226517-a2192b80-85b7-11eb-8dba-c01399fb7105.png" width="350"/> <img src="https://user-images.githubusercontent.com/12687085/111226510-a0e7fe80-85b7-11eb-913b-09776cdaa92e.png" width="230"/>  
<!-- ![basis B](https://user-images.githubusercontent.com/12687085/111226517-a2192b80-85b7-11eb-8dba-c01399fb7105.png "title2")
![basis A](https://user-images.githubusercontent.com/12687085/111226510-a0e7fe80-85b7-11eb-913b-09776cdaa92e.png "title1") -->


Checkout our [documentation](https://emlp.readthedocs.io/en/latest/) to see how to use our system and some worked examples.


# Installation instructions

To install as a package, run `pip install emlp`.

To run the scripts you will instead need to clone the repo and install it locally which you can do with
```bash
git clone https://github.com/mfinzi/equivariant-MLP.git
cd equivariant-MLP
pip install -e .
```

# Experimental Results from Paper

Assuming you have installed the repo locally, you can run the experiments we described in the paper. 

To train the regression models on one of the `Inertia`, `O5Synthetic`, or `ParticleInteraction` datasets found in [`emlp.datasets.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/datasets.py) you can run the script [`experiments/train_regression.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/experiments/train_regression.py) with command line arguments specifying the dataset, network, and symmetry group. For example to train [`EMLP`](https://emlp.readthedocs.io/en/latest/package/emlp.nn.html#emlp.nn.EMLP) with [`SO(3)`](https://emlp.readthedocs.io/en/latest/package/emlp.groups.html#emlp.groups.SO) equivariance on the `Inertia` dataset, you can run

```
python experiments/train_regression.py --dataset Inertia --network EMLP --group "SO(3)"
```

or to train the MLP baseline you can run

```
python experiments/train_regression.py --dataset Inertia --network MLP
```
Other command line arguments such as `--aug=True` for data augmentation or `--ch=512` for number of hidden units and others are available, and you can browse the options and their defaults with `python experiments/train_regression.py -h`. If no group is specified, EMLP will automatically choose the one matched to the dataset, but you can also go crazy with any of the other groups implemented in [`groups.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/groups.py) provided the dimensions match the data (e.g. for the 3D inertia dataset you could do `--group=` [`"Z(3)"`](https://emlp.readthedocs.io/en/latest/package/emlp.groups.html#emlp.groups.Z) or [`"DkeR3(3)"`](https://emlp.readthedocs.io/en/latest/package/emlp.groups.html#emlp.groups.DkeR3) but not [`"Sp(2)"`](https://emlp.readthedocs.io/en/latest/package/emlp.groups.html#emlp.groups.Sp) or [`"SU(5)"`](https://emlp.readthedocs.io/en/latest/package/emlp.groups.html#emlp.groups.SU)).

For the dynamical systems modeling experiments you can use the scripts
[`experiments/neuralode.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/experiments/neuralode.py) to train (equivariant) Neural ODEs and [`experiments/hnn.py`](https://github.com/mfinzi/equivariant-MLP/blob/master/experiments/hnn.py) to train (equivariant) Hamiltonian Neural Networks.


For the dynamical system task, the Neural ODE and HNN models have special names. [`EMLPode`](https://emlp.readthedocs.io/en/latest/package/emlp.nn.html#emlp.nn.EMLPode) and [`MLPode`](https://emlp.readthedocs.io/en/latest/package/emlp.nn.html#emlp.nn.MLPode) for the Neural ODEs in `neuralode.py` and [`EMLPH`](https://emlp.readthedocs.io/en/latest/package/emlp.nn.html#emlp.nn.EMLPH) and [`MLPH`](https://emlp.readthedocs.io/en/latest/package/emlp.nn.html#emlp.nn.MLPH) for the HNNs in `hnn.py`. For example,

```
python experiments/neuralode.py --network EMLPode --group="O2eR3()"
```
or 

```
python experiments/hnn.py --network EMLPH --group="DkeR3(6)"
```

These models are trained to fit a double spring dynamical system. 30s rollouts of the dataset, along with rollout error on these trajectories, and conservation of angular momentum are shown below.

<img src="https://user-images.githubusercontent.com/12687085/114937183-759d3d00-9e0b-11eb-9310-bbfc606e6bda.gif" width="250"/> <img src="https://user-images.githubusercontent.com/12687085/114937167-703ff280-9e0b-11eb-8421-d8408b31908a.PNG" width="300"/> <img src="https://user-images.githubusercontent.com/12687085/114937171-71711f80-9e0b-11eb-885e-a541ae1d28cc.PNG" width="260"/> 

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
