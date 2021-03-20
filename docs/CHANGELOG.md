# Change Log

<!---
This is a comment.
Remember to align the itemized text with the first line of an item within a list.
-->

## EMLP 0.8.0 (Unreleased)

* New features:
  * Fallback autograd jvp implementation of drho to make implementing new reps easier.
  * Mixed group representations (now working and tested)
  * Experimental support of complex groups and representations
* Bug Fixes:
  * Element ordering of mixed groups is now correctly maintained in the solution
  * Fixed edge case of {func}`lazy_direct_matmat` when concatenating matrices of size 0 
    affecting {func}`emlp.solver.representation.Rep.symmetric_basis` but not 
    {func}`emlp.solver.representation.Rep.symmetric_projector`
