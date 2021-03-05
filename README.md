# A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups

--------------------------------------------------------------------------------
**[Documentation](https://emlp.readthedocs.io/en/latest/)** | **[Paper]()**| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfinzi/equivariant-MLP/blob/master/emlp.ipynb)


*EMLP* is a jax library for the automated construction of equivariant layers in deep learning.

--------------------------------------------------------------------------------

# Installation instructions

To install as a package, run `pip install git+https://github.com/mfinzi/equivariant-MLP.git`.

To run the scripts you will instead need to clone the repo and install it locally which you can do with
```bash
git clone https://github.com/mfinzi/equivariant-MLP.git
cd equivariant-MLP
pip install -e .
```

<!-- # Train Models
We have implemented a variety of challenging benchmarks for modeling physical dynamical systems such as ``ChainPendulum``, ``CoupledPendulum``,``MagnetPendulum``,``Gyroscope``,``Rotor`` which can be selected with the ``--body-class`` argument.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94081999-eb897a80-fdcd-11ea-8e29-c676d4e25f64.PNG" width=1000>
</p>

You can run our models ``CHNN`` and ``CLNN`` as well as the baseline ``NN`` (NeuralODE), ``DeLaN``, and ``HNN`` models with the ``network-class`` argument as shown below.

```
python pl_trainer.py --network-class CHNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class CLNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class HNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class DeLaN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class NN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
```

Our explicitly constrained ``CHNN`` and ``CLNN`` outperform the competing methods by several orders of magnitude across the different benchmarks as shown below.
<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94081992-e75d5d00-fdcd-11ea-9df0-576af6909944.PNG" width=1000>
</p> -->

If you find our work helpful, please cite it with
```bibtex
@article{finzi2021arbitrary,
  title={A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups},
  author={Finzi, Marc and Welling, Max and Wilson, Andrew Gordon},
  journal={Arxiv},
  year={2021}
}
```
<!-- 
Top quark tagging dataset: https://zenodo.org/record/2603256#.YAoEPehKiUl -->
