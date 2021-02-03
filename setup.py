from setuptools import setup,find_packages
import sys, os

setup(name="Equivariant-MLP",
      description="Generic Equivariant Multi Layer Perceptrons using Tensor Types",
      version='0.1',
      author='Marc Finzi',
      author_email='maf820@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py','tables','objax',
      'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml','optax',
      'torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq'],#'objax'
      packages=find_packages(),
      long_description=open('README.md').read(),
)