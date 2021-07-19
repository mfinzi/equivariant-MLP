# Change Log

<!---
This is a comment.
Remember to align the itemized text with the first line of an item within a list.
-->

## EMLP 1.0.0
* New Features
  * Flax support (see `using EMLP with Flax`)
  * Auto generated `size()`, `__eq__`, `__hash__`, and `.T` methods for new representations
  * You can now use ints in place of `Scalars` for direct sum, e.g. add `3+V`
* Codebase improvements
  * Streamlined product_sum_reps direct sum and product rules, now with plumb dispatch
  * More general `Dual(Rep)` implementation that now works with any kind of Rep, not just `V`
  * CI setup and with more tests

## EMLP 0.9.0
* Cross Platform Support:
  * You can now use EMLP in PyTorch, check out `Using EMLP with PyTorch`
  * You can also use EMLP with Haiku in jax, check out `Using EMLP with Haiku`

* Bug Fixes
  * Fixed broken constraints with Trivial group

## EMLP 0.8.0 (Unreleased)

* New features:
  * Fallback autograd jvp implementation of drho to make implementing new reps easier.
  * Mixed group representations (now working and tested)
  * Experimental support of complex groups and representations
* Bug Fixes:
  * Element ordering of mixed groups is now correctly maintained in the solution
  * Fixed edge case of {func}`lazy_direct_matmat` when concatenating matrices of size 0 
    affecting {func}`emlp.reps.Rep.equivariant_basis` but not 
    {func}`emlp.reps.Rep.equivariant_projector`
* API Changes:
  * `emlp.solver.representation` -> `emlp.reps`
  * `emlp.solver.groups` -> `emlp.groups`
  * `emlp.models.mlp` -> `emlp.nn`
  * `rep.symmetric_basis()` -> `rep.equivariant_basis()`
  * `rep.symmetric_projector()` -> `rep.equivariant_projector()`
  * Tests and experiments separated from package and api

