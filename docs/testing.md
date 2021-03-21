# Running the Tests

We have a number of tests checking the equivariance of representations constructed in
 different ways (`.T`, `*`, `+`) for the groups that have been implemented (`Z(n)`,`S(n)`,`D(k)`,`SO(n)`, `O(n)`,`Sp(n)`,`SO13()`,`O13()`,`SU(n)`).
We use pytest and some of the tests are automatically generated. Because there is a large amount of tests and it can take quite some time to run them all (about 15 minutes),
you can run a subset using pytests built in features to filter by the matches on the name of the testcase using the `-k` argument.

For example to run `test_prod` with all the groups you can run
```pytest tests/equivariance_tests.py -k "test_prod"```

To run the test case for a specific group could use the filter `-k "test_prod and SO3"` and to run all tests with that group
you could run
```pytest tests/equivariance_tests.py -k "SO3"```

Due to pytest parsing limitations, all parenthesis in the test names are stripped.
To list all available tests, (or those that match certain `-k` arguments) use `--co` (for collect only).

The usual pytest command line arguments apply (like `-v` for verbose). 
 <!-- and we add an additional `--log` argument for the log level. -->

Similarly, you can find tests for "mixed" representations containing sub-representations from different groups in `emlp/tests/product_groups_tests.py`.