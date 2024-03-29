# PULP-NNX testing

## Repository structure

- app: test application
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

## Application

The Makefile in the `app/` uses a flag `ACCELERATOR` to decide which accelerator to use.
The choices are _ne16_ or _neureka_.
You can either export it or run it like `ACCELERATOR=<accelerator> make clean all run`.
