.. EMLP documentation master file, created by
   sphinx-quickstart on Mon Feb 22 18:41:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


EMLP reference documentation
============================

A type system for the automated construction of equivariant layers. 
EMLP is designed to make constructing equivariant layers with different matrix groups
and representations an easy task, and one that does not require knowledge of analytic solutions.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   notebooks/1quickstart.ipynb
   notebooks/2building_a_model.ipynb
   notebooks/3new_groups.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   notebooks/4new_representations.ipynb
   notebooks/5mixed_tensors.ipynb
   notebooks/6multilinear_maps.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Cross Platform Support

   notebooks/pytorch_support.ipynb
   notebooks/haiku_support.ipynb
   notebooks/flax_support.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference

   package/emlp.groups
   package/emlp.reps
   package/emlp.nn

.. toctree::
   :maxdepth: 1
   :caption: Notes

   testing.md
   inner_workings.md
   CHANGELOG.md

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   documentation.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
