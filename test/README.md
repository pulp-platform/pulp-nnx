# PULP-NNX testing

## Repository structure

- app_*: test application
    - inc: test application headers
    - src: test application sources
    - gen_inc: generated test headers for data and test information
- tests: passing tests

## Usage

We use `pytest` to run all of our tests defined in [test.py](test.py).

Parameters that have to be defined:

- `--test-dir` (`-T`): path to the test directory, can be set multiple times

Optional parameters

- `--recursive` (`-R`): recursively search the given test directories for tests

**Example**: Run all tests in *tests*
```
$ pytest test.py --test-dir tests --recursive
```

For more information you can run
```
$ pytest test.py --help
```

## Helper scripts

- [testgen.py](testgen.py): collection of helper tools for individual tests

For more information you can run the script with the `-h` flag.

## Applications

There are 2 applications provided, one testing the more abstract pulp_nnx interface and another one just the hardware abstraction layer, called `app_pulp_nnx` and `app_pulp_nnx_hal`.
For information on the testing applications and how to build them, take a look in their dedicated readmes: [app_pulp_nnx/README.md](app_pulp_nnx/README.md) and [app_pulp_nnx_hal/README.md](app_pulp_nnx_hal/README.md).
