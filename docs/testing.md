# Running the Tests

We have a number of tests checking the equivariance of representations constructed in
 different ways (`.T`, `*`, `+`) for the groups that have been implemented (`Z(n)`,`S(n)`,`D(k)`,`SO(n)`, `O(n)`,`Sp(n)`,`SO13()`,`O13()`,`SU(n)`).
We use pytest and some of the tests are automatically generated. Because there is a large amount of tests and it can take quite some time to run them all,
you can run a subset using pytests built in features to filter by the matches on the name of the testcase using the `-k` argument.

For example to run `test_prod` with all the groups you can run
```python emlp/tests/equivariant_subspaces_tests.py -k "test_prod"```

To run the test case for a specific group could use the filter `-k "test_prod_O(3)"` and to run all tests with that group
you could run
```python emlp/tests/equivariant_subspaces_tests.py -k "O(3)"```

The usual pytest command line arguments apply (like `-v` for verbose) and we add an additional `--log` argument for the log level.