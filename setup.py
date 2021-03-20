from setuptools import setup,find_packages
import sys, os

setup(name="Equivariant-MLP",
      description="A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups",
      version='0.8',
      author='Marc Finzi',
      author_email='maf820@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py','objax','pytest',
      'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml','optax','tqdm>=4.38'],
      packages=find_packages(),
      long_description=open('README.md').read(),
)